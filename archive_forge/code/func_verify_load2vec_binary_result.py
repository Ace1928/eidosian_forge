import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def verify_load2vec_binary_result(self, w2v_dict, binary_chunk_size, limit):
    tmpfile = gensim.test.utils.get_tmpfile('tmp_w2v')
    save_dict_to_word2vec_formated_file(tmpfile, w2v_dict)
    w2v_model = gensim.models.keyedvectors._load_word2vec_format(cls=gensim.models.KeyedVectors, fname=tmpfile, binary=True, limit=limit, binary_chunk_size=binary_chunk_size)
    if limit is None:
        limit = len(w2v_dict)
    w2v_keys_postprocessed = list(w2v_dict.keys())[:limit]
    w2v_dict_postprocessed = {k.lstrip(): w2v_dict[k] for k in w2v_keys_postprocessed}
    self.assert_dict_equal_to_model(w2v_dict_postprocessed, w2v_model)