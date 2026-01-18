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
def test_similarities(self):
    """Test similarity and n_similarity methods."""
    model = word2vec.Word2Vec(vector_size=2, min_count=1, sg=0, hs=0, negative=2)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    self.assertTrue(model.wv.n_similarity(['graph', 'trees'], ['trees', 'graph']))
    self.assertTrue(model.wv.n_similarity(['graph'], ['trees']) == model.wv.similarity('graph', 'trees'))
    self.assertRaises(ZeroDivisionError, model.wv.n_similarity, ['graph', 'trees'], [])
    self.assertRaises(ZeroDivisionError, model.wv.n_similarity, [], ['graph', 'trees'])
    self.assertRaises(ZeroDivisionError, model.wv.n_similarity, [], [])