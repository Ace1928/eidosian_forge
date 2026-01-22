import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class UsesDictionary(BaseAnalyzer):
    """A BaseAnalyzer that uses a Dictionary, hence can translate tokens to counts.
    The standard BaseAnalyzer can only deal with token ids since it doesn't have the token2id
    mapping.

    Attributes
    ----------
    relevant_words : set
        Set of words that occurrences should be accumulated for.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        Dictionary based on text
    token2id : dict
        Mapping from :class:`~gensim.corpora.dictionary.Dictionary`

    """

    def __init__(self, relevant_ids, dictionary):
        """

        Parameters
        ----------
        relevant_ids : dict
            Mapping
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Dictionary based on text

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.topic_coherence import text_analysis
            >>> from gensim.corpora.dictionary import Dictionary
            >>>
            >>> ids = {1: 'foo', 2: 'bar'}
            >>> dictionary = Dictionary([['foo', 'bar', 'baz'], ['foo', 'bar', 'bar', 'baz']])
            >>> udict = text_analysis.UsesDictionary(ids, dictionary)
            >>>
            >>> print(udict.relevant_words)
            set([u'foo', u'baz'])

        """
        super(UsesDictionary, self).__init__(relevant_ids)
        self.relevant_words = _ids_to_words(self.relevant_ids, dictionary)
        self.dictionary = dictionary
        self.token2id = dictionary.token2id

    def get_occurrences(self, word):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self._get_occurrences(self.id2contiguous[word_id])

    def _word2_contiguous_id(self, word):
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self.id2contiguous[word_id]

    def get_co_occurrences(self, word1, word2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        word_id1 = self._word2_contiguous_id(word1)
        word_id2 = self._word2_contiguous_id(word2)
        return self._get_co_occurrences(word_id1, word_id2)