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
class NativeTrainingContinuationTest(unittest.TestCase):
    maxDiff = None
    model_structural_sanity = TestFastTextModel.model_structural_sanity

    def setUp(self):
        expected = {u'quick': [0.023393, 0.11499, 0.11684, -0.13349, 0.022543], u'brown': [0.015288, 0.050404, -0.041395, -0.090371, 0.06441], u'fox': [0.061692, 0.082914, 0.020081, -0.039159, 0.03296], u'jumps': [0.070107, 0.081465, 0.051763, 0.012084, 0.0050402], u'over': [0.055023, 0.03465, 0.01648, -0.11129, 0.094555], u'lazy': [-0.022103, -0.020126, -0.033612, -0.049473, 0.0054174], u'dog': [0.084983, 0.09216, 0.020204, -0.13616, 0.01118]}
        self.oov_expected = {word: np.array(arr, dtype=np.float32) for word, arr in expected.items()}

    def test_in_vocab(self):
        """Test for correct representation of in-vocab words."""
        native = load_native()
        with utils.open(datapath('toy-model.vec'), 'r', encoding='utf-8') as fin:
            expected = dict(load_vec(fin))
        for word, expected_vector in expected.items():
            actual_vector = native.wv.get_vector(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-05))
        self.model_structural_sanity(native)

    def test_out_of_vocab(self):
        """Test for correct representation of out-of-vocab words."""
        native = load_native()
        for word, expected_vector in self.oov_expected.items():
            actual_vector = native.wv.get_vector(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-05))
        self.model_structural_sanity(native)

    def test_sanity(self):
        """Compare models trained on toy data.  They should be equal."""
        trained = train_gensim()
        native = load_native()
        self.assertEqual(trained.wv.bucket, native.wv.bucket)
        compare_wv(trained.wv, native.wv, self)
        compare_vocabulary(trained, native, self)
        compare_nn(trained, native, self)
        self.model_structural_sanity(trained)
        self.model_structural_sanity(native)

    def test_continuation_native(self):
        """Ensure that training has had a measurable effect."""
        native = load_native()
        self.model_structural_sanity(native)
        word = 'society'
        old_vector = native.wv.get_vector(word).tolist()
        native.train(list_corpus, total_examples=len(list_corpus), epochs=native.epochs)
        new_vector = native.wv.get_vector(word).tolist()
        self.assertNotEqual(old_vector, new_vector)
        self.model_structural_sanity(native)

    def test_continuation_gensim(self):
        """Ensure that continued training has had a measurable effect."""
        model = train_gensim(min_count=0)
        self.model_structural_sanity(model)
        vectors_ngrams_before = np.copy(model.wv.vectors_ngrams)
        word = 'human'
        old_vector = model.wv.get_vector(word).tolist()
        model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)
        vectors_ngrams_after = np.copy(model.wv.vectors_ngrams)
        self.assertFalse(np.allclose(vectors_ngrams_before, vectors_ngrams_after))
        new_vector = model.wv.get_vector(word).tolist()
        self.assertNotEqual(old_vector, new_vector)
        self.model_structural_sanity(model)

    def test_save_load_gensim(self):
        """Test that serialization works end-to-end.  Not crashing is a success."""
        model_name = 'test_ft_saveload_native.model'
        with temporary_file(model_name):
            train_gensim().save(model_name)
            model = FT_gensim.load(model_name)
            self.model_structural_sanity(model)
            model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)
            model.save(model_name)
            self.model_structural_sanity(model)

    def test_save_load_native(self):
        """Test that serialization works end-to-end.  Not crashing is a success."""
        model_name = 'test_ft_saveload_fb.model'
        with temporary_file(model_name):
            load_native().save(model_name)
            model = FT_gensim.load(model_name)
            self.model_structural_sanity(model)
            model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)
            model.save(model_name)
            self.model_structural_sanity(model)

    def test_load_native_pretrained(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('toy-model-pretrained.bin'))
        actual = model.wv['monarchist']
        expected = np.array([0.76222, 1.0669, 0.7055, -0.090969, -0.53508])
        self.assertTrue(np.allclose(expected, actual, atol=0.001))
        self.model_structural_sanity(model)

    def test_load_native_vectors(self):
        cap_path = datapath('crime-and-punishment.bin')
        fbkv = gensim.models.fasttext.load_facebook_vectors(cap_path)
        self.assertFalse('landlord' in fbkv.key_to_index)
        self.assertTrue('landlady' in fbkv.key_to_index)
        oov_vector = fbkv['landlord']
        iv_vector = fbkv['landlady']
        self.assertFalse(np.allclose(oov_vector, iv_vector))

    def test_no_ngrams(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('crime-and-punishment.bin'))
        v1 = model.wv['']
        origin = np.zeros(v1.shape, v1.dtype)
        self.assertTrue(np.allclose(v1, origin))
        self.model_structural_sanity(model)