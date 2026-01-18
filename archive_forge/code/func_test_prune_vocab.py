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
def test_prune_vocab(self):
    """Test Prune vocab while scanning sentences"""
    sentences = [['graph', 'system'], ['graph', 'system'], ['system', 'eps'], ['graph', 'system']]
    model = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, max_vocab_size=2, seed=42, hs=1, negative=0)
    self.assertEqual(len(model.wv), 2)
    self.assertEqual(model.wv.get_vecattr('graph', 'count'), 3)
    self.assertEqual(model.wv.get_vecattr('system', 'count'), 4)
    sentences = [['graph', 'system'], ['graph', 'system'], ['system', 'eps'], ['graph', 'system'], ['minors', 'survey', 'minors', 'survey', 'minors']]
    model = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, max_vocab_size=2, seed=42, hs=1, negative=0)
    self.assertEqual(len(model.wv), 3)
    self.assertEqual(model.wv.get_vecattr('graph', 'count'), 3)
    self.assertEqual(model.wv.get_vecattr('minors', 'count'), 3)
    self.assertEqual(model.wv.get_vecattr('system', 'count'), 4)