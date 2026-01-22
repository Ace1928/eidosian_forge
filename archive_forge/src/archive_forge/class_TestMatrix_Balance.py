import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
class TestMatrix_Balance:

    def test_string_arg(self):
        assert_raises(ValueError, matrix_balance, 'Some string for fail')

    def test_infnan_arg(self):
        assert_raises(ValueError, matrix_balance, np.array([[1, 2], [3, np.inf]]))
        assert_raises(ValueError, matrix_balance, np.array([[1, 2], [3, np.nan]]))

    def test_scaling(self):
        _, y = matrix_balance(np.array([[1000, 1], [1000, 0]]))
        assert_allclose(np.diff(np.log2(np.diag(y))), [5])

    def test_scaling_order(self):
        A = np.array([[1, 0, 0.0001], [1, 1, 0.01], [10000.0, 100.0, 1]])
        x, y = matrix_balance(A)
        assert_allclose(solve(y, A).dot(y), x)

    def test_separate(self):
        _, (y, z) = matrix_balance(np.array([[1000, 1], [1000, 0]]), separate=1)
        assert_equal(np.diff(np.log2(y)), [5])
        assert_allclose(z, np.arange(2))

    def test_permutation(self):
        A = block_diag(np.ones((2, 2)), np.tril(np.ones((2, 2))), np.ones((3, 3)))
        x, (y, z) = matrix_balance(A, separate=1)
        assert_allclose(y, np.ones_like(y))
        assert_allclose(z, np.array([0, 1, 6, 5, 4, 3, 2]))

    def test_perm_and_scaling(self):
        cases = (np.array([[0.0, 0.0, 0.0, 0.0, 2e-06], [0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2e-06, 0.0, 0.0]]), np.array([[-0.5, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, -0.5, 0.0], [0.0, 1.0, 0.0, -1.0]]), np.array([[-3.0, 0.0, 1.0, 0.0], [-1.0, -1.0, -0.0, 1.0], [-3.0, -0.0, -0.0, 0.0], [-1.0, -0.0, 1.0, -1.0]]))
        for A in cases:
            x, y = matrix_balance(A)
            x, (s, p) = matrix_balance(A, separate=1)
            ip = np.empty_like(p)
            ip[p] = np.arange(A.shape[0])
            assert_allclose(y, np.diag(s)[ip, :])
            assert_allclose(solve(y, A).dot(y), x)