import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class InvertedIndexAccumulator(WindowedTextsAnalyzer, InvertedIndexBased):
    """Build an inverted index from a sequence of corpus texts."""

    def analyze_text(self, window, doc_num=None):
        for word_id in window:
            if word_id is not self._none_token:
                self._inverted_index[word_id].add(self._num_docs)