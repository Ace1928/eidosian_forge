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
def test_rule_with_min_count(self):
    """Test that returning RULE_DEFAULT from trim_rule triggers min_count."""
    model = word2vec.Word2Vec(sentences + [['occurs_only_once']], min_count=2, trim_rule=_rule)
    self.assertTrue('human' not in model.wv)
    self.assertTrue('occurs_only_once' not in model.wv)
    self.assertTrue('interface' in model.wv)