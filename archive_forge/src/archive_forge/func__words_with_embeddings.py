import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def _words_with_embeddings(self, ids):
    if not hasattr(ids, '__iter__'):
        ids = [ids]
    words = [self.dictionary.id2token[word_id] for word_id in ids]
    return [word for word in words if word in self.model]