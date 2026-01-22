import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def _slide_window(self, window, doc_num):
    if doc_num != self._current_doc_num:
        self._uniq_words[:] = False
        self._uniq_words[np.unique(window)] = True
        self._current_doc_num = doc_num
    else:
        self._uniq_words[self._token_at_edge] = False
        self._uniq_words[window[-1]] = True
    self._token_at_edge = window[0]