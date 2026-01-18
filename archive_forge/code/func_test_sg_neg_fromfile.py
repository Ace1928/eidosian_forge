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
def test_sg_neg_fromfile(self):
    model = word2vec.Word2Vec(sg=1, window=4, hs=0, negative=15, min_count=5, epochs=10, workers=2)
    self.model_sanity(model, with_corpus_file=True)