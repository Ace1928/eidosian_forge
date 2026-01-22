import numpy as np
from numpy.testing import (assert_allclose,
import scipy.linalg.cython_blas as blas
class TestDGEMM:

    def test_transposes(self):
        a = np.arange(12, dtype='d').reshape((3, 4))[:2, :2]
        b = np.arange(1, 13, dtype='d').reshape((4, 3))[:2, :2]
        c = np.empty((2, 4))[:2, :2]
        blas._test_dgemm(1.0, a, b, 0.0, c)
        assert_allclose(c, a.dot(b))
        blas._test_dgemm(1.0, a.T, b, 0.0, c)
        assert_allclose(c, a.T.dot(b))
        blas._test_dgemm(1.0, a, b.T, 0.0, c)
        assert_allclose(c, a.dot(b.T))
        blas._test_dgemm(1.0, a.T, b.T, 0.0, c)
        assert_allclose(c, a.T.dot(b.T))
        blas._test_dgemm(1.0, a, b, 0.0, c.T)
        assert_allclose(c, a.dot(b).T)
        blas._test_dgemm(1.0, a.T, b, 0.0, c.T)
        assert_allclose(c, a.T.dot(b).T)
        blas._test_dgemm(1.0, a, b.T, 0.0, c.T)
        assert_allclose(c, a.dot(b.T).T)
        blas._test_dgemm(1.0, a.T, b.T, 0.0, c.T)
        assert_allclose(c, a.T.dot(b.T).T)

    def test_shapes(self):
        a = np.arange(6, dtype='d').reshape((3, 2))
        b = np.arange(-6, 2, dtype='d').reshape((2, 4))
        c = np.empty((3, 4))
        blas._test_dgemm(1.0, a, b, 0.0, c)
        assert_allclose(c, a.dot(b))
        blas._test_dgemm(1.0, b.T, a.T, 0.0, c.T)
        assert_allclose(c, b.T.dot(a.T).T)