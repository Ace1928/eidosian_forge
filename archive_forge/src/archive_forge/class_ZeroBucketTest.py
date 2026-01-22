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
class ZeroBucketTest(unittest.TestCase):
    """Test FastText with no buckets / no-ngrams: essentially FastText-as-Word2Vec."""

    def test_in_vocab(self):
        model = train_gensim(bucket=0)
        self.assertIsNotNone(model.wv['anarchist'])

    def test_out_of_vocab(self):
        model = train_gensim(bucket=0)
        with self.assertRaises(KeyError):
            model.wv.get_vector('streamtrain')

    def test_cbow_neg(self):
        """See `gensim.test.test_word2vec.TestWord2VecModel.test_cbow_neg`."""
        model = FT_gensim(sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=15, min_count=5, epochs=10, workers=2, sample=0, max_n=0)
        TestWord2VecModel.model_sanity(self, model)