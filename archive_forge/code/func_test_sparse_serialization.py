import os
import unittest
import random
import shutil
import numpy as np
from scipy import sparse
from gensim.utils import is_corpus, mock_data
from gensim.corpora.sharded_corpus import ShardedCorpus
def test_sparse_serialization(self):
    no_exception = True
    try:
        ShardedCorpus(self.tmp_fname, self.data, shardsize=100, dim=self.dim, sparse_serialization=True)
    except Exception:
        no_exception = False
        raise
    finally:
        self.assertTrue(no_exception)