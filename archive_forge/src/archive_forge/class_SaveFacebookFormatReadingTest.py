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
class SaveFacebookFormatReadingTest(unittest.TestCase):
    """
    This class containts tests that check the following scenario:

    + create fastText model using gensim
    + save file to model.bin
    + retrieve word vectors from model.bin using fasttext Facebook utility
    + compare vectors retrieved by Facebook utility with those obtained directly from gensim model
    """

    def _check_load_fasttext_format(self, sg):
        model_params = {'sg': sg, 'vector_size': 10, 'min_count': 1, 'hs': 1, 'negative': 5, 'bucket': BUCKET, 'seed': 42, 'workers': 1}
        with temporary_file('load_fasttext.bin') as fpath:
            model = _create_and_save_fb_model(fpath, model_params)
            wv = _read_wordvectors_using_fasttext(fpath, model.wv.index_to_key)
        for i, w in enumerate(model.wv.index_to_key):
            diff = calc_max_diff(wv[i, :], model.wv[w])
            self.assertLess(diff, 0.0001)

    def test_skipgram(self):
        self._check_load_fasttext_format(sg=1)

    def test_cbow(self):
        self._check_load_fasttext_format(sg=0)