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
def test_online_learning(self):
    """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
    model_hs = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, seed=42, hs=1, negative=0)
    model_neg = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, seed=42, hs=0, negative=5)
    self.assertTrue(len(model_hs.wv), 12)
    self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 3)
    model_hs.build_vocab(new_sentences, update=True)
    model_neg.build_vocab(new_sentences, update=True)
    self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 4)
    self.assertTrue(model_hs.wv.get_vecattr('artificial', 'count'), 4)
    self.assertEqual(len(model_hs.wv), 14)
    self.assertEqual(len(model_neg.wv), 14)