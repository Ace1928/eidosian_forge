from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
def tfidf_sort_key(term_index):
    if isinstance(term_index, tuple):
        term_index, *_ = term_index
    term_idf = tfidf.idfs[term_index]
    return (-term_idf, term_index)