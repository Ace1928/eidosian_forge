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
def test_path_line_sentences_one_file(self):
    """Does PathLineSentences work with a single file argument?"""
    test_file = os.path.join(datapath('PathLineSentences'), '1.txt')
    with utils.open(test_file, 'rb') as orig:
        sentences = word2vec.PathLineSentences(test_file)
        for words in sentences:
            self.assertEqual(words, utils.to_unicode(orig.readline()).split())