import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_similarity(self):
    """Test similarity returns expected value for two words, and for identical words."""
    self.assertTrue(np.allclose(self.vectors.similarity('war', 'war'), 1))
    self.assertTrue(np.allclose(self.vectors.similarity('war', 'conflict'), 0.93305397))