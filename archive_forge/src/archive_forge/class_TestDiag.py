from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestDiag:

    def test_vector(self):
        vals = (100 * arange(5)).astype('l')
        b = zeros((5, 5))
        for k in range(5):
            b[k, k] = vals[k]
        assert_equal(diag(vals), b)
        b = zeros((7, 7))
        c = b.copy()
        for k in range(5):
            b[k, k + 2] = vals[k]
            c[k + 2, k] = vals[k]
        assert_equal(diag(vals, k=2), b)
        assert_equal(diag(vals, k=-2), c)

    def test_matrix(self, vals=None):
        if vals is None:
            vals = (100 * get_mat(5) + 1).astype('l')
        b = zeros((5,))
        for k in range(5):
            b[k] = vals[k, k]
        assert_equal(diag(vals), b)
        b = b * 0
        for k in range(3):
            b[k] = vals[k, k + 2]
        assert_equal(diag(vals, 2), b[:3])
        for k in range(3):
            b[k] = vals[k + 2, k]
        assert_equal(diag(vals, -2), b[:3])

    def test_fortran_order(self):
        vals = array(100 * get_mat(5) + 1, order='F', dtype='l')
        self.test_matrix(vals)

    def test_diag_bounds(self):
        A = [[1, 2], [3, 4], [5, 6]]
        assert_equal(diag(A, k=2), [])
        assert_equal(diag(A, k=1), [2])
        assert_equal(diag(A, k=0), [1, 4])
        assert_equal(diag(A, k=-1), [3, 6])
        assert_equal(diag(A, k=-2), [5])
        assert_equal(diag(A, k=-3), [])

    def test_failure(self):
        assert_raises(ValueError, diag, [[[1]]])