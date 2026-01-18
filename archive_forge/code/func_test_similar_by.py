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
def test_similar_by(self):
    """Test word2vec similar_by_word and similar_by_vector."""
    model = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)
    wordsims = model.wv.similar_by_word('graph', topn=10)
    wordsims2 = model.wv.most_similar(positive='graph', topn=10)
    vectorsims = model.wv.similar_by_vector(model.wv['graph'], topn=10)
    vectorsims2 = model.wv.most_similar([model.wv['graph']], topn=10)
    self.assertEqual(wordsims, wordsims2)
    self.assertEqual(vectorsims, vectorsims2)