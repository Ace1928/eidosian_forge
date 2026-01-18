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
def test_train_with_explicit_param(self):
    model = word2vec.Word2Vec(vector_size=2, min_count=1, hs=1, negative=0)
    model.build_vocab(sentences)
    with self.assertRaises(ValueError):
        model.train(sentences, total_examples=model.corpus_count)
    with self.assertRaises(ValueError):
        model.train(sentences, epochs=model.epochs)
    with self.assertRaises(ValueError):
        model.train(sentences)