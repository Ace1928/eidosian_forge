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
@unittest.skipIf(sys.maxunicode == 65535, "Python interpreter doesn't support UCS-4 (wide unicode)")
def test_text_cy_wide_unicode(self):
    for word in self.expected_text_wide_unicode:
        expected = self.expected_text_wide_unicode[word]
        actual = compute_ngrams(word, 3, 5)
        self.assertEqual(expected, actual)