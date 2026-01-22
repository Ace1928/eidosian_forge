import logging
import pickle
import random
from collections import defaultdict
from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI
def _load_data_conll_format(filename):
    print('Read from file: ', filename)
    with open(filename, 'rb') as fin:
        sentences = []
        sentence = []
        for line in fin.readlines():
            line = line.strip()
            if len(line) == 0:
                sentences.append(sentence)
                sentence = []
                continue
            tokens = line.split('\t')
            word = tokens[1]
            tag = tokens[4]
            sentence.append((word, tag))
        return sentences