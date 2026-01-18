from __future__ import print_function
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from filelock import FileLock
import os
import argparse
import tarfile
import numpy as np
import re
from ray import train, tune
def vectorize_stories(word_idx, story_maxlen, query_maxlen, data):
    inputs, queries, answers = ([], [], [])
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen), pad_sequences(queries, maxlen=query_maxlen), np.array(answers))