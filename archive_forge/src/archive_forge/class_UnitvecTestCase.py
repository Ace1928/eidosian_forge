import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
class UnitvecTestCase(unittest.TestCase):

    def test_sparse_npfloat32(self):
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.float32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_sparse_npfloat64(self):
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.float64)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_sparse_npint32(self):
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.int32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_sparse_npint64(self):
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.int64)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_dense_npfloat32(self):
        input_vector = np.random.uniform(size=(5,)).astype(np.float32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_dense_npfloat64(self):
        input_vector = np.random.uniform(size=(5,)).astype(np.float64)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_dense_npint32(self):
        input_vector = np.random.randint(10, size=5).astype(np.int32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_dense_npint64(self):
        input_vector = np.random.randint(10, size=5).astype(np.int32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_sparse_python_float(self):
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(float)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_sparse_python_int(self):
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(int)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_dense_python_float(self):
        input_vector = np.random.uniform(size=(5,)).astype(float)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_dense_python_int(self):
        input_vector = np.random.randint(10, size=5).astype(int)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_return_norm_zero_vector_scipy_sparse(self):
        input_vector = sparse.csr_matrix([[]], dtype=np.int32)
        return_value = matutils.unitvec(input_vector, return_norm=True)
        self.assertTrue(isinstance(return_value, tuple))
        norm = return_value[1]
        self.assertTrue(isinstance(norm, float))
        self.assertEqual(norm, 1.0)

    def test_return_norm_zero_vector_numpy(self):
        input_vector = np.array([], dtype=np.int32)
        return_value = matutils.unitvec(input_vector, return_norm=True)
        self.assertTrue(isinstance(return_value, tuple))
        norm = return_value[1]
        self.assertTrue(isinstance(norm, float))
        self.assertEqual(norm, 1.0)

    def test_return_norm_zero_vector_gensim_sparse(self):
        input_vector = []
        return_value = matutils.unitvec(input_vector, return_norm=True)
        self.assertTrue(isinstance(return_value, tuple))
        norm = return_value[1]
        self.assertTrue(isinstance(norm, float))
        self.assertEqual(norm, 1.0)