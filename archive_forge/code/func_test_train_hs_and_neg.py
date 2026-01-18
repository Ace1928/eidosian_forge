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
@log_capture()
def test_train_hs_and_neg(self, loglines):
    """
        Test if ValueError is raised when both hs=0 and negative=0
        Test if warning is raised if both hs and negative are activated
        """
    with self.assertRaises(ValueError):
        word2vec.Word2Vec(sentences, min_count=1, hs=0, negative=0)
    word2vec.Word2Vec(sentences, min_count=1, hs=1, negative=5)
    warning = 'Both hierarchical softmax and negative sampling are activated.'
    self.assertTrue(warning in str(loglines))