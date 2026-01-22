import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
class Gensim320Test(unittest.TestCase):

    def test(self):
        path = datapath('old_keyedvectors_320.dat')
        vectors = gensim.models.keyedvectors.KeyedVectors.load(path)
        self.assertTrue(vectors.get_vector('computer') is not None)