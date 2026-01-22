import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class WindowedTextsAnalyzer(UsesDictionary):
    """Gather some stats about relevant terms of a corpus by iterating over windows of texts."""

    def __init__(self, relevant_ids, dictionary):
        """

        Parameters
        ----------
        relevant_ids : set of int
            Relevant id
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Dictionary instance with mappings for the relevant_ids.

        """
        super(WindowedTextsAnalyzer, self).__init__(relevant_ids, dictionary)
        self._none_token = self._vocab_size

    def accumulate(self, texts, window_size):
        relevant_texts = self._iter_texts(texts)
        windows = utils.iter_windows(relevant_texts, window_size, ignore_below_size=False, include_doc_num=True)
        for doc_num, virtual_document in windows:
            if len(virtual_document) > 0:
                self.analyze_text(virtual_document, doc_num)
            self.num_docs += 1
        return self

    def _iter_texts(self, texts):
        dtype = np.uint16 if np.iinfo(np.uint16).max >= self._vocab_size else np.uint32
        for text in texts:
            ids = (self.id2contiguous[self.token2id[w]] if w in self.relevant_words else self._none_token for w in text)
            yield np.fromiter(ids, dtype=dtype, count=len(text))