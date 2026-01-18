import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_return_norm_zero_vector_numpy(self):
    input_vector = np.array([], dtype=np.int32)
    return_value = matutils.unitvec(input_vector, return_norm=True)
    self.assertTrue(isinstance(return_value, tuple))
    norm = return_value[1]
    self.assertTrue(isinstance(norm, float))
    self.assertEqual(norm, 1.0)