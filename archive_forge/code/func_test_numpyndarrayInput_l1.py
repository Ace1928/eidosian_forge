import logging
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath, get_tmpfile
def test_numpyndarrayInput_l1(self):
    """Test for np ndarray input for l1 transformation"""
    ndarray_matrix = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
    normalized = self.model_l1.normalize(ndarray_matrix)
    self.assertTrue(isinstance(normalized, np.ndarray))
    expected = np.array([[0.04761905, 0.0, 0.0952381], [0.0, 0.0, 0.14285714], [0.19047619, 0.23809524, 0.28571429]])
    self.assertTrue(np.allclose(normalized, expected))
    self.assertRaises(ValueError, lambda model, doc: model.normalize(doc), self.model_l1, [1, 2, 3])