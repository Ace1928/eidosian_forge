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
def test_sg_hs_online(self):
    """Test skipgram w/ hierarchical softmax"""
    model = word2vec.Word2Vec(sg=1, window=5, hs=1, negative=0, min_count=3, epochs=10, seed=42, workers=2)
    self.onlineSanity(model)