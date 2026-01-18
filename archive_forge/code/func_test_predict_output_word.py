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
def test_predict_output_word(self):
    """Test word2vec predict_output_word method handling for negative sampling scheme"""
    model_with_neg = word2vec.Word2Vec(sentences, min_count=1)
    predictions_with_neg = model_with_neg.predict_output_word(['system', 'human'], topn=5)
    self.assertTrue(len(predictions_with_neg) == 5)
    predictions_out_of_vocab = model_with_neg.predict_output_word(['some', 'random', 'words'], topn=5)
    self.assertEqual(predictions_out_of_vocab, None)
    tmpf = get_tmpfile('gensim_word2vec.tst')
    model_with_neg.wv.save_word2vec_format(tmpf, binary=True)
    kv_model_with_neg = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=True)
    binary_model_with_neg = word2vec.Word2Vec()
    binary_model_with_neg.wv = kv_model_with_neg
    self.assertRaises(RuntimeError, binary_model_with_neg.predict_output_word, ['system', 'human'])
    model_without_neg = word2vec.Word2Vec(sentences, min_count=1, hs=1, negative=0)
    self.assertRaises(RuntimeError, model_without_neg.predict_output_word, ['system', 'human'])
    str_context = ['system', 'human']
    mixed_context = [model_with_neg.wv.get_index(str_context[0]), str_context[1]]
    idx_context = [model_with_neg.wv.get_index(w) for w in str_context]
    prediction_from_str = model_with_neg.predict_output_word(str_context, topn=5)
    prediction_from_mixed = model_with_neg.predict_output_word(mixed_context, topn=5)
    prediction_from_idx = model_with_neg.predict_output_word(idx_context, topn=5)
    self.assertEqual(prediction_from_str, prediction_from_mixed)
    self.assertEqual(prediction_from_str, prediction_from_idx)