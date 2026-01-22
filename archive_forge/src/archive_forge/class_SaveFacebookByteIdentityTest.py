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
@unittest.skipIf(not FT_CMD, 'fasttext not in FT_HOME or PATH, skipping test')
class SaveFacebookByteIdentityTest(unittest.TestCase):
    """
    This class containts tests that check the following scenario:

    + create binary fastText file model1.bin using facebook_binary (FT)
    + load file model1.bin to variable `model`
    + save `model` to model2.bin using gensim
    + check if files model1.bin and model2.bin are byte-identical
    """

    def _check_roundtrip_file_file(self, sg):
        model_params = {'vector_size': 10, 'sg': sg, 'seed': 42}
        with temporary_file('m1.bin') as m1, temporary_file('m2.bin') as m2, temporary_file('m1.vec'):
            m1_basename = m1[:-4]
            _save_test_model(m1_basename, model_params)
            model = gensim.models.fasttext.load_facebook_model(m1)
            gensim.models.fasttext.save_facebook_model(model, m2)
            bin1 = _read_binary_file(m1)
            bin2 = _read_binary_file(m2)
        self.assertEqual(bin1, bin2)

    def test_skipgram(self):
        self._check_roundtrip_file_file(sg=1)

    def test_cbow(self):
        self._check_roundtrip_file_file(sg=0)