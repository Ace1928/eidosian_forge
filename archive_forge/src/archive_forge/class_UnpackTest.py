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
class UnpackTest(unittest.TestCase):

    def test_sanity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {10: 0, 11: 1, 12: 2}
        n = _unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[10]))
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[11]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[12]))

    def test_tricky(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {1: 0, 0: 1, 12: 2}
        n = _unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[0]))
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[1]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[12]))

    def test_identity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {0: 0, 1: 1, 2: 2}
        n = _unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[0]))
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[1]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[2]))