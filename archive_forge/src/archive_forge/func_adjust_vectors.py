import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def adjust_vectors(self):
    """Adjust the vectors for words in the vocabulary.

        The adjustment composes the trained full-word-token vectors with
        the vectors of the subword ngrams, matching the Facebook reference
        implementation behavior.

        """
    if self.bucket == 0:
        self.vectors = self.vectors_vocab
        return
    self.vectors = self.vectors_vocab[:].copy()
    for i, _ in enumerate(self.index_to_key):
        ngram_buckets = self.buckets_word[i]
        for nh in ngram_buckets:
            self.vectors[i] += self.vectors_ngrams[nh]
        self.vectors[i] /= len(ngram_buckets) + 1