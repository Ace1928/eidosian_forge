import json
import os
from os.path import join as pjoin
from os.path import isfile, isdir
from parlai.core.params import ParlaiParser
from time import time
from data_utils import sentence_split, tf_idf_vec, tf_idf_dist

File adapted from
https://github.com/facebookresearch/ELI5/blob/master/data_creation/select_sentences_tfidf.py
Modified to use data directory rather than a hard-coded processed data directory
