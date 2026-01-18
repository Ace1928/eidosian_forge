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
def test_online_learning_after_save(self):
    """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
    tmpf = get_tmpfile('gensim_word2vec.tst')
    model_neg = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, seed=42, hs=0, negative=5)
    model_neg.save(tmpf)
    model_neg = word2vec.Word2Vec.load(tmpf)
    self.assertTrue(len(model_neg.wv), 12)
    model_neg.build_vocab(new_sentences, update=True)
    model_neg.train(new_sentences, total_examples=model_neg.corpus_count, epochs=model_neg.epochs)
    self.assertEqual(len(model_neg.wv), 14)