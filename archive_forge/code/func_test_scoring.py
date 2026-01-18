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
def test_scoring(self):
    """Test word2vec scoring."""
    model = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)
    scores = model.score(sentences, len(sentences))
    self.assertEqual(len(scores), len(sentences))