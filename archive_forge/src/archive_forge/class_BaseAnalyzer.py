import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class BaseAnalyzer:
    """Base class for corpus and text analyzers.

    Attributes
    ----------
    relevant_ids : dict
        Mapping
    _vocab_size : int
        Size of vocabulary.
    id2contiguous : dict
        Mapping word_id -> number.
    log_every : int
        Interval for logging.
    _num_docs : int
        Number of documents.

    """

    def __init__(self, relevant_ids):
        """

        Parameters
        ----------
        relevant_ids : dict
            Mapping

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.topic_coherence import text_analysis
            >>> ids = {1: 'fake', 4: 'cats'}
            >>> base = text_analysis.BaseAnalyzer(ids)
            >>> # should return {1: 'fake', 4: 'cats'} 2 {1: 0, 4: 1} 1000 0
            >>> print(base.relevant_ids, base._vocab_size, base.id2contiguous, base.log_every, base._num_docs)
            {1: 'fake', 4: 'cats'} 2 {1: 0, 4: 1} 1000 0

        """
        self.relevant_ids = relevant_ids
        self._vocab_size = len(self.relevant_ids)
        self.id2contiguous = {word_id: n for n, word_id in enumerate(self.relevant_ids)}
        self.log_every = 1000
        self._num_docs = 0

    @property
    def num_docs(self):
        return self._num_docs

    @num_docs.setter
    def num_docs(self, num):
        self._num_docs = num
        if self._num_docs % self.log_every == 0:
            logger.info('%s accumulated stats from %d documents', self.__class__.__name__, self._num_docs)

    def analyze_text(self, text, doc_num=None):
        raise NotImplementedError('Base classes should implement analyze_text.')

    def __getitem__(self, word_or_words):
        if isinstance(word_or_words, str) or not hasattr(word_or_words, '__iter__'):
            return self.get_occurrences(word_or_words)
        else:
            return self.get_co_occurrences(*word_or_words)

    def get_occurrences(self, word_id):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        return self._get_occurrences(self.id2contiguous[word_id])

    def _get_occurrences(self, word_id):
        raise NotImplementedError('Base classes should implement occurrences')

    def get_co_occurrences(self, word_id1, word_id2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        return self._get_co_occurrences(self.id2contiguous[word_id1], self.id2contiguous[word_id2])

    def _get_co_occurrences(self, word_id1, word_id2):
        raise NotImplementedError('Base classes should implement co_occurrences')