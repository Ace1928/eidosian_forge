import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_save_reload(self):
    randkv = KeyedVectors(vector_size=100)
    count = 20
    keys = [str(i) for i in range(count)]
    weights = [pseudorandom_weak_vector(randkv.vector_size) for _ in range(count)]
    randkv.add_vectors(keys, weights)
    tmpfiletxt = gensim.test.utils.get_tmpfile('tmp_kv.txt')
    randkv.save_word2vec_format(tmpfiletxt, binary=False)
    reloadtxtkv = KeyedVectors.load_word2vec_format(tmpfiletxt, binary=False)
    self.assertEqual(randkv.index_to_key, reloadtxtkv.index_to_key)
    self.assertTrue((randkv.vectors == reloadtxtkv.vectors).all())
    tmpfilebin = gensim.test.utils.get_tmpfile('tmp_kv.bin')
    randkv.save_word2vec_format(tmpfilebin, binary=True)
    reloadbinkv = KeyedVectors.load_word2vec_format(tmpfilebin, binary=True)
    self.assertEqual(randkv.index_to_key, reloadbinkv.index_to_key)
    self.assertTrue((randkv.vectors == reloadbinkv.vectors).all())