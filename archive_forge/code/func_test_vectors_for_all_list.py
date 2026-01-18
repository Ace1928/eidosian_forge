import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_vectors_for_all_list(self):
    """Test vectors_for_all returns expected results with a list of keys."""
    words = ['conflict', 'administration', 'terrorism', 'an out-of-vocabulary word', 'another out-of-vocabulary word']
    vectors_for_all = self.vectors.vectors_for_all(words)
    expected = 3
    predicted = len(vectors_for_all)
    assert expected == predicted
    expected = self.vectors['conflict']
    predicted = vectors_for_all['conflict']
    assert np.allclose(expected, predicted)