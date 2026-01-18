import logging
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath, get_tmpfile
def test_tupleInput_l2(self):
    """Test tuple input for l2 transformation"""
    normalized = self.model_l2.normalize(self.doc)
    expected = [(1, 0.4082482904638631), (5, 0.8164965809277261), (8, 0.4082482904638631)]
    self.assertTrue(np.allclose(normalized, expected))