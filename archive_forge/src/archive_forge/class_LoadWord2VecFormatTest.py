import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
class LoadWord2VecFormatTest(unittest.TestCase):

    def assert_dict_equal_to_model(self, d, m):
        self.assertEqual(len(d), len(m))
        for word in d.keys():
            self.assertSequenceEqual(list(d[word]), list(m[word]))

    def verify_load2vec_binary_result(self, w2v_dict, binary_chunk_size, limit):
        tmpfile = gensim.test.utils.get_tmpfile('tmp_w2v')
        save_dict_to_word2vec_formated_file(tmpfile, w2v_dict)
        w2v_model = gensim.models.keyedvectors._load_word2vec_format(cls=gensim.models.KeyedVectors, fname=tmpfile, binary=True, limit=limit, binary_chunk_size=binary_chunk_size)
        if limit is None:
            limit = len(w2v_dict)
        w2v_keys_postprocessed = list(w2v_dict.keys())[:limit]
        w2v_dict_postprocessed = {k.lstrip(): w2v_dict[k] for k in w2v_keys_postprocessed}
        self.assert_dict_equal_to_model(w2v_dict_postprocessed, w2v_model)

    def test_load_word2vec_format_basic(self):
        w2v_dict = {'abc': [1, 2, 3], 'cde': [4, 5, 6], 'def': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=None)
        w2v_dict = {'abc': [1, 2, 3], 'cdefg': [4, 5, 6], 'd': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=None)

    def test_load_word2vec_format_limit(self):
        w2v_dict = {'abc': [1, 2, 3], 'cde': [4, 5, 6], 'def': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)
        w2v_dict = {'abc': [1, 2, 3], 'cde': [4, 5, 6], 'def': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)
        w2v_dict = {'abc': [1, 2, 3], 'cdefg': [4, 5, 6], 'd': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)
        w2v_dict = {'abc': [1, 2, 3], 'cdefg': [4, 5, 6], 'd': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)

    def test_load_word2vec_format_space_stripping(self):
        w2v_dict = {'\nabc': [1, 2, 3], 'cdefdg': [4, 5, 6], '\n\ndef': [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)