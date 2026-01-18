import logging
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath, get_tmpfile
def test_tupleInput_l1(self):
    """Test tuple input for l1 transformation"""
    normalized = self.model_l1.normalize(self.doc)
    expected = [(1, 0.25), (5, 0.5), (8, 0.25)]
    self.assertTrue(np.allclose(normalized, expected))