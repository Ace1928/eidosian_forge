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
def test_r_n_g(self):
    """Test word2vec results identical with identical RNG seed."""
    model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
    model2 = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
    self.models_equal(model, model2)