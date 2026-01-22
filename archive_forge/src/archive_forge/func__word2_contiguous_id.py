import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def _word2_contiguous_id(self, word):
    try:
        word_id = self.token2id[word]
    except KeyError:
        word_id = word
    return self.id2contiguous[word_id]