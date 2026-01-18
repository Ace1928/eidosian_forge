import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_vocab(self):
    """Test word2vec vocabulary building."""
    corpus = LeeCorpus()
    total_words = sum((len(sentence) for sentence in corpus))
    model = word2vec.Word2Vec(min_count=1, hs=1, negative=0)
    model.build_vocab(corpus)
    self.assertTrue(len(model.wv) == 6981)
    self.assertEqual(sum((model.wv.get_vecattr(k, 'count') for k in model.wv.key_to_index)), total_words)
    np.allclose(model.wv.get_vecattr('the', 'code'), [1, 1, 0, 0])
    model = word2vec.Word2Vec(hs=1, negative=0)
    model.build_vocab(corpus)
    self.assertTrue(len(model.wv) == 1750)
    np.allclose(model.wv.get_vecattr('the', 'code'), [1, 1, 1, 0])
    self.assertRaises(RuntimeError, word2vec.Word2Vec, [])
    self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words + 1)