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
def test_rule(self):
    """Test applying vocab trim_rule to build_vocab instead of constructor."""
    model = word2vec.Word2Vec(min_count=1)
    model.build_vocab(sentences, trim_rule=_rule)
    self.assertTrue('human' not in model.wv)