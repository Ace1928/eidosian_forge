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
def test_word2vec_stand_alone_script(self):
    """Does Word2Vec script launch standalone?"""
    cmd = [sys.executable, '-m', 'gensim.scripts.word2vec_standalone', '-train', datapath('testcorpus.txt'), '-output', 'vec.txt', '-size', '200', '-sample', '1e-4', '-binary', '0', '-iter', '3', '-min_count', '1']
    output = check_output(args=cmd, stderr=subprocess.PIPE)
    self.assertEqual(output, b'')