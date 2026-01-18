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
def test_oov_similarity(self):
    word = 'someoovword'
    most_similar = self.test_model.wv.most_similar(word)
    top_neighbor, top_similarity = most_similar[0]
    v1 = self.test_model.wv[word]
    v2 = self.test_model.wv[top_neighbor]
    top_similarity_direct = self.test_model.wv.cosine_similarities(v1, v2.reshape(1, -1))[0]
    self.assertAlmostEqual(top_similarity, top_similarity_direct, places=6)