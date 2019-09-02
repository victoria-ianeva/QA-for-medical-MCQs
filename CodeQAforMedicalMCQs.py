
# coding: utf-8

# In[ ]:

import keras
import h5py
import numpy as np
import time
from keras import initializers, regularizers, constraints
from keras import backend as K


# In[ ]:

import pickle
word_index=pickle.load(open("word_index.pkl","rb"))


# In[ ]:

def load_mesh(filename):
    file=open(filename,"r", encoding="utf-8")
    state=0
    results={}
    entries=[]
    all_entries={}
    reversed_entries={}
    for line in file:
        
        if(line.startswith("*NEWRECORD")):
            state=1
            entries=[]
            continue
        if(state==1):
            if(line.startswith("MH = ")):
                heading=line.strip()[5:]
            if(line.startswith("UI = ")):
                d=line.strip()[5:]
            if(line.startswith("ENTRY = ")):
                entries.append(line.strip()[8:].split("|")[0])
            if(line.strip()==""):
                results[d.lower()]=heading
                entries.append(heading)
                all_entries[d.lower()]=entries
                reversed_entries[heading.lower()]=[d.lower()]
                for entry in entries:
                    if (not(entry in reversed_entries.keys())):
                        reversed_entries[entry.lower()]=[]
                    reversed_entries[entry.lower()].append(d.lower())
                        
                state=0
    return results,all_entries,reversed_entries
        
d_h,entries,reversed_entries=load_mesh("d2017.bin")


# In[ ]:

from keras.preprocessing import text
import xml.etree.ElementTree as ET

file_xml=open("usmle_step1_step2_sample.xml","r")
#file_xml=open("H:\\answerresponse\\usmle_step1_step2_sample.xml","r")
tree = ET.parse(file_xml)#"items_usmle_samples.xml")
root=tree.getroot()
correct=0
total=0
total_single=0
item_h=0
total_distractors=0
Machine_Predictions={}
#all_d=set()
texts=[]
correct_answers=[]
distractors_all=[]
correct_answers_texts=[]
IDS=[]
correct_answers_all=[]
texts_all=[]
distractors_items=[]
IDS_ALL=[]
for item in root.iter("Item"):
    Options=item.find("Options")
    Text=item.find("ItemText")
    ID=item.find("MedleyID")
    distractors=[]
    distractors_text=[]
    if(Text==None):
        continue
    #print (ID.text)
    #print(Text.text)
    if(Options!=None):
        for s in Options.iter("string"):
            #print(s.text)
            if(s.text.lower() in reversed_entries.keys()):
                #print (reversed_entries[s.text.lower()])
                distractors.append(reversed_entries[s.text.lower()][0])
                #all_d |=set(reversed_entries[s.text.lower()])
            distractors_text.append(s.text)
    leadin=item.find("LeadIn")
    if(leadin!=None):
        leadin=leadin.text
    else:
        leadin=""
    if(leadin.startswith("5 mEq")):
        continue
    CorrectAnswers=item.find("CorrectAnswers")
    if(CorrectAnswers!=None):
        distractors_items.append(distractors_text)
        correct_answers_all.append(CorrectAnswers.text)
        texts_all.append(Text.text+" "+leadin)
        IDS_ALL.append(ID.text)
    if(len(distractors)>1):
        item_h += 1
    #print (distractors)
    if(CorrectAnswers!=None) and CorrectAnswers.text.lower() in reversed_entries.keys():
        
        texts.append(Text.text+" "+leadin)
        correct_answers.append(reversed_entries[CorrectAnswers.text.lower()][0])
        distractors_all.append(distractors)
        correct_answers_texts.append(CorrectAnswers.text.lower())
        IDS.append(ID.text)

word_sequences=[[word_index[word] if (word in word_index) and word_index[word]<200000 else 0 for word in ts[0:250]]+[0]*(250-len(ts)) 
                 for ts in [keras.preprocessing.text.text_to_word_sequence(item_text,lower=False) for item_text in texts]]
word_sequences_all=[[word_index[word] if (word in word_index) and word_index[word]<200000 else 0 for word in ts][0:250]+[0]*(250-min(250,len(ts))) 
                 for ts in [keras.preprocessing.text.text_to_word_sequence(item_text,lower=False) for item_text in texts_all]]
distractors_sequences=[[[word_index[word] if (word in word_index) and word_index[word]<200000 else 0 for word in ts]+[0]*(250-len(ts)) 
                 for ts in [keras.preprocessing.text.text_to_word_sequence(distractor,lower=False) for distractor in distractors]]
for distractors in distractors_items]

distractors_sequences_flatten=[x for distractors in distractors_sequences for x in distractors]
distractors_lens=[len(x) for x in distractors_items]
distractors_index=[0]*len(distractors_lens)
for i in range(1,len(distractors_index)):
    distractors_index[i]=distractors_index[i-1]+distractors_lens[i-1]


# In[ ]:

def unigram(s):
    return ((w,) for w in s)
from nltk import bigrams,trigrams
distractors_mesh_trigrams={}
for i in range(len(distractors_items)):
    distractors_mesh_trigrams[IDS_ALL[i]]=[]
    dx={}
    for d in distractors_items[i]:
        dx[d]=set()
        for gramsfunction in [unigram,bigrams,trigrams]:
            grams = gramsfunction(d.split())
            for gram in grams:
                if(" ".join(gram).lower() in reversed_entries):
                    #print(gram,reversed_entries[" ".join(gram).lower()][0])
                    dx[d]=dx[d].union(reversed_entries[" ".join(gram).lower()])
    distractors_mesh_trigrams[IDS_ALL[i]]=dx
    #print(IDS_ALL[i])
focus_d={d:i for (i,d) in enumerate(sorted(set([mesh for i in range(len(IDS_ALL)) for d in distractors_mesh_trigrams[IDS_ALL[i]] 
     for mesh in distractors_mesh_trigrams[IDS_ALL[i]][d]]).union(set([d for distractors in distractors_items 
             for distractor in distractors if distractor.lower() in reversed_entries
                 for d in reversed_entries[distractor.lower()]]))))}


# In[ ]:

f= h5py.File('H:/preprocessing_medline/myfile_int_1_1000_.hdf5','r')
heading_vocab_size=len(f["medline_heading_index"])

heading_vocab_size=len(focus_d)
def generator_input_output_fn(dataset_in,dataset_out,loop=True,batch_size=1000):
    i=0
    while 1:
        
        inputs = []
        outputs = []

        for abstract_int,heading_int in zip(dataset_in,dataset_out):
            one_hot=one_hot_focus(heading_int,old_new_map,heading_vocab_size)
            if(sum(one_hot)==0):
                continue
            if(all([x==0 for x in abstract_int])):
                continue
            inputs.append([x if x<200000 else 0 for x in abstract_int])
            #inputs.append(abstract_int)
            #outputs.append(one_hot(heading_int,heading_vocab_size))
            outputs.append(one_hot)
            if(i==batch_size):

                yield np.array(inputs),np.array(outputs)
                inputs=[]
                outputs=[]
                i=0
            else:
                i += 1
        yield np.array(inputs),np.array(outputs)
        print("1 run")
        if not (loop):
            break
def one_hot(data,num):
    
    x=[0.0]*num
    for i in data:
        if(i in old_new_map):
            try:
                x[old_new_map[i]]=1.0
            except:
                print(data)
    return x
def one_hot_all(data,num):
    
    x=[0.0]*num
    for i in data:
        if(i<num):
            x[i]=1
    return x
def one_hot_focus(data,old_new_map,num):
    
    x=[0.0]*num
    for i in data:
        if(i in old_new_map):
            x[old_new_map[i]]=1
    return x


# In[ ]:

class EvaluateEvery(keras.callbacks.Callback):
    def __init__(self,num_epoch,x_y_test_generator,model,BOW=False,correctThreshold=600,prefix=""):
        self.num_epoch=num_epoch
        self.x_y_test_generator=x_y_test_generator
        self.model=model
        self.corrects=[]
        self.corrects_sim=[]
        self.f_measures=[]
        self.BOW=BOW
        self.correctThreshold=correctThreshold
        self.prefix=prefix
    def on_train_begin(self, logs={}):
        self.epoch_count=0

    def on_epoch_end(self, batch, logs={}):
        self.epoch_count += 1
        #if(self.epoch_count%self.num_epoch==0):
        if (False):
            total_predicted=0
            total_label=0
            total_correct=0
            i=0
            print (self.epoch_count)
            for (x_test,y_test) in self.x_y_test_generator:
                predictions=model.predict_on_batch(x_test)
                predicted,label,correct=evaluate(predictions,y_test,0.41)
                total_predicted += predicted
                total_label += label
                total_correct += correct
                i += 1
                if(i%10==0):
                    print (total_correct,total_predicted,total_label)
                    break
        if(self.epoch_count%(self.num_epoch)==0):

            #correct,correct_sim=extrinsic_evaluation(self.model,self.BOW)
            correct,correct_sim,_=extrinsic_evaluation_all_mesh(self.model)#,self.BOW)
            print(correct,correct_sim)
            self.corrects.append(correct)
            self.corrects_sim.append(correct_sim)
            
            #self.f_measures.append(total_correct*2/(total_predicted+total_label))
            


# In[ ]:

def extrinsic_evaluation_all_mesh(model):
    predictions_texts_all=model.predict(np.array(word_sequences_all))
    five_d=0
    correct=0
    results={}
    aggregatefunctions={"max":max
                        ,"sum":sum
                        ,"mean":lambda x: sum(x)/len(x)
                        ,"min":min}
    corrects={aggre:0 for aggre in aggregatefunctions}
    for i in range(0,len(predictions_texts_all)):
        text_predictions=predictions_texts_all[i]
        mesh_predictions=[]
        correct_prediction_values={}
        for d in distractors_mesh_trigrams[IDS_ALL[i]]:
            
            if(len([h for h in distractors_mesh_trigrams[IDS_ALL[i]][d] if h in old_indices])>0):
                my_prediction=max([text_predictions[old_new_map[old_indices[h]]]
                                   for h in distractors_mesh_trigrams[IDS_ALL[i]][d] if h in old_indices])
                
                my_predictions={aggregate:aggregatefunctions[aggregate]
                                ([text_predictions[old_new_map[old_indices[h]]]
                                   for h in distractors_mesh_trigrams[IDS_ALL[i]][d] if h in old_indices]) 
                                for aggregate in aggregatefunctions}
                mesh_predictions.append(my_predictions)
                if(d==correct_answers_all[i]):
                    correct_prediction_values=my_predictions
        
        if(True
           and (len(distractors_mesh_trigrams[IDS_ALL[i]][correct_answers_all[i]])>0)
           and (len([d for d in distractors_mesh_trigrams[IDS_ALL[i]] 
                     if len([h for h in distractors_mesh_trigrams[IDS_ALL[i]][d] if h in old_indices])>0])==5)):
            
            results[IDS_ALL[i]]={}
            for aggreatefunc in aggregatefunctions:
                max_prediction=max([x[aggreatefunc] for x in mesh_predictions])
                
                if((correct_prediction_values[aggreatefunc]==max_prediction)
                    and len([x[aggreatefunc] for x in mesh_predictions if x[aggreatefunc]==max_prediction])==1):
                    corrects[aggreatefunc]+=1
                    results[IDS_ALL[i]][aggreatefunc]=1
                #if((correct_prediction_values[aggreatefunc]==max_prediction)
                #    and len([x[aggreatefunc] for x in mesh_predictions if x[aggreatefunc]==max_prediction])!=1):
                    #print([x[aggreatefunc] for x in mesh_predictions])
            five_d+=1
    print("five distractors ngram mesh:{}, correct: {}".format(five_d,corrects))
    return corrects["mean"],five_d,results


# In[ ]:

old_indices={d:old_index for old_index,d in enumerate(f["medline_heading_index"])}
old_new_map={i:focus_d[d] for (i,d) in enumerate(f["medline_heading_index"]) if d in focus_d}
count_headings=np.load("count_headings_new_all_1000.npy")
weight_headings={old_new_map[i]:1000000/float(x) if x != 0 else 0 for (i,x) in enumerate(count_headings) if i in old_new_map}

""" Attention code from https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043 """
class Attention(keras.layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


# In[ ]:

inputs_layer=keras.layers.Input(shape=(250,))
embedding_layer=keras.layers.Embedding(200000,256,input_length=250)(inputs_layer)
#batch_norm=keras.layers.BatchNormalization()(embedding_layer)
conv1Ds3=keras.layers.Conv1D(500,
                 3,
                 padding='same',
                 activation='relu',
                 strides=1)(embedding_layer)
conv1Ds4=keras.layers.Conv1D(500,
                 4,
                 padding='same',
                 activation='relu',
                 strides=1)(embedding_layer)
conv1Ds2=keras.layers.Conv1D(500,
                 2,
                 padding='same',
                 activation='relu',
                 strides=1)(embedding_layer)
units=32
#GlobalMaxPooling1D_l=keras.layers.GlobalAveragePooling1D()(keras.layers.Concatenate()([conv1Ds3,conv1Ds2,conv1Ds4]))
attentions=Attention(250)(keras.layers.Concatenate()([conv1Ds3,conv1Ds2,conv1Ds4]))
#batch_norm=keras.layers.Dropout(0.2)(GlobalMaxPooling1D_l)
denses=keras.layers.Dense(heading_vocab_size,activation="softmax")(attentions)
model_conv1d=keras.models.Model(inputs=inputs_layer,outputs=denses)
model_conv1d.compile(optimizer='Nadam',
          loss='binary_crossentropy',
                 metrics=['acc'])
model_conv1d.summary()
lstm=keras.layers.Bidirectional(keras.layers.LSTM(units,return_sequences=True))(embedding_layer)
flatten=keras.layers.Flatten()(lstm)
#dropout=keras.layers.Dropout(0.5)(flatten)
denses2=keras.layers.Dense(heading_vocab_size,activation="softmax")(flatten)
model_lstm=keras.models.Model(inputs=inputs_layer,outputs=denses2)
model_lstm.compile(optimizer='Nadam',
          loss='binary_crossentropy',
                 metrics=['acc'])
model_lstm.summary()


# In[ ]:

for kkk in range(0,10):
    generator_input_output_train=generator_input_output_fn(f["medline_abstract_int_train"]
                                                          ,f["medline_heading_int_train"],False,128)
    

    
    model.fit_generator(generator_input_output_train,steps_per_epoch=400
                        ,epochs=200
                        , class_weight=weight_headings)
    
    keras.models.save_model(model,"epoch_"
                            +str(kkk)+
                            +"_" +str(time.time()))


# In[ ]:


                            


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



