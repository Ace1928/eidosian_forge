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
class SaveGensimByteIdentityTest(unittest.TestCase):
    """
    This class containts tests that check the following scenario:

    + create binary fastText file model1.bin using gensim
    + load file model1.bin to variable `model`
    + save `model` to model2.bin
    + check if files model1.bin and model2.bin are byte identical
    """

    def _check_roundtrip_file_file(self, sg):
        model_params = {'sg': sg, 'vector_size': 10, 'min_count': 1, 'hs': 1, 'negative': 0, 'bucket': BUCKET, 'seed': 42, 'workers': 1}
        with temporary_file('roundtrip_file_to_file1.bin') as fpath1, temporary_file('roundtrip_file_to_file2.bin') as fpath2:
            _create_and_save_fb_model(fpath1, model_params)
            model = gensim.models.fasttext.load_facebook_model(fpath1)
            gensim.models.fasttext.save_facebook_model(model, fpath2)
            bin1 = _read_binary_file(fpath1)
            bin2 = _read_binary_file(fpath2)
        self.assertEqual(bin1, bin2)

    def test_skipgram(self):
        self._check_roundtrip_file_file(sg=1)

    def test_cbow(self):
        self._check_roundtrip_file_file(sg=0)