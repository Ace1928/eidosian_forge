import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_vectors_for_all_with_copy_vecattrs(self):
    """Test vectors_for_all returns can copy vector attributes."""
    words = ['conflict']
    vectors_for_all = self.vectors.vectors_for_all(words, copy_vecattrs=True)
    expected = self.vectors.get_vecattr('conflict', 'count')
    predicted = vectors_for_all.get_vecattr('conflict', 'count')
    assert expected == predicted