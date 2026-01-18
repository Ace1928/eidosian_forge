import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_sparse_npfloat32(self):
    input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.float32)
    unit_vector = matutils.unitvec(input_vector)
    man_unit_vector = manual_unitvec(input_vector)
    self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
    self.assertEqual(input_vector.dtype, unit_vector.dtype)