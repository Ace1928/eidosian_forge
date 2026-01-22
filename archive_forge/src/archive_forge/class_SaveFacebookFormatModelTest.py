from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
class SaveFacebookFormatModelTest(unittest.TestCase):

    def _check_roundtrip(self, sg):
        model_params = {'sg': sg, 'vector_size': 10, 'min_count': 1, 'hs': 1, 'negative': 5, 'seed': 42, 'bucket': BUCKET, 'workers': 1}
        with temporary_file('roundtrip_model_to_model.bin') as fpath:
            model_trained = _create_and_save_fb_model(fpath, model_params)
            model_loaded = gensim.models.fasttext.load_facebook_model(fpath)
        self.assertEqual(model_trained.vector_size, model_loaded.vector_size)
        self.assertEqual(model_trained.window, model_loaded.window)
        self.assertEqual(model_trained.epochs, model_loaded.epochs)
        self.assertEqual(model_trained.negative, model_loaded.negative)
        self.assertEqual(model_trained.hs, model_loaded.hs)
        self.assertEqual(model_trained.sg, model_loaded.sg)
        self.assertEqual(model_trained.wv.bucket, model_loaded.wv.bucket)
        self.assertEqual(model_trained.wv.min_n, model_loaded.wv.min_n)
        self.assertEqual(model_trained.wv.max_n, model_loaded.wv.max_n)
        self.assertEqual(model_trained.sample, model_loaded.sample)
        self.assertEqual(set(model_trained.wv.index_to_key), set(model_loaded.wv.index_to_key))
        for w in model_trained.wv.index_to_key:
            v_orig = model_trained.wv[w]
            v_loaded = model_loaded.wv[w]
            self.assertLess(calc_max_diff(v_orig, v_loaded), MAX_WORDVEC_COMPONENT_DIFFERENCE)

    def test_skipgram(self):
        self._check_roundtrip(sg=1)

    def test_cbow(self):
        self._check_roundtrip(sg=0)