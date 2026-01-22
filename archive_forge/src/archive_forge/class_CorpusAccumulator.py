import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class CorpusAccumulator(InvertedIndexBased):
    """Gather word occurrence stats from a corpus by iterating over its BoW representation."""

    def analyze_text(self, text, doc_num=None):
        """Build an inverted index from a sequence of corpus texts."""
        doc_words = frozenset((x[0] for x in text))
        top_ids_in_doc = self.relevant_ids.intersection(doc_words)
        for word_id in top_ids_in_doc:
            self._inverted_index[self.id2contiguous[word_id]].add(self._num_docs)

    def accumulate(self, corpus):
        for document in corpus:
            self.analyze_text(document)
            self.num_docs += 1
        return self