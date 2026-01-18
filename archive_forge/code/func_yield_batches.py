import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def yield_batches(self, texts):
    """Return a generator over the given texts that yields batches of `batch_size` texts at a time."""
    batch = []
    for text in self._iter_texts(texts):
        batch.append(text)
        if len(batch) == self.batch_size:
            yield batch
            batch = []
    if batch:
        yield batch