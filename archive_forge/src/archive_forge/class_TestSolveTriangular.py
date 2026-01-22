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
class TestSolveTriangular:

    def test_simple(self):
        """
        solve_triangular on a simple 2x2 matrix.
        """
        A = array([[1, 0], [1, 2]])
        b = [1, 1]
        sol = solve_triangular(A, b, lower=True)
        assert_array_almost_equal(sol, [1, 0])
        sol = solve_triangular(A.T, b, lower=False)
        assert_array_almost_equal(sol, [0.5, 0.5])
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [0.5, 0.5])
        b = identity(2)
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [[1.0, -0.5], [0, 0.5]])

    def test_simple_complex(self):
        """
        solve_triangular on a simple 2x2 complex matrix
        """
        A = array([[1 + 1j, 0], [1j, 2]])
        b = identity(2)
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [[0.5 - 0.5j, -0.25 - 0.25j], [0, 0.5]])
        b = np.diag([1 + 1j, 1 + 2j])
        sol = solve_triangular(A, b, lower=True, trans=0)
        assert_array_almost_equal(sol, [[1, 0], [-0.5j, 0.5 + 1j]])
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [[1, 0.25 - 0.75j], [0, 0.5 + 1j]])
        sol = solve_triangular(A, b, lower=True, trans=2)
        assert_array_almost_equal(sol, [[1j, -0.75 - 0.25j], [0, 0.5 + 1j]])
        sol = solve_triangular(A.T, b, lower=False, trans=0)
        assert_array_almost_equal(sol, [[1, 0.25 - 0.75j], [0, 0.5 + 1j]])
        sol = solve_triangular(A.T, b, lower=False, trans=1)
        assert_array_almost_equal(sol, [[1, 0], [-0.5j, 0.5 + 1j]])
        sol = solve_triangular(A.T, b, lower=False, trans=2)
        assert_array_almost_equal(sol, [[1j, 0], [-0.5, 0.5 + 1j]])

    def test_check_finite(self):
        """
        solve_triangular on a simple 2x2 matrix.
        """
        A = array([[1, 0], [1, 2]])
        b = [1, 1]
        sol = solve_triangular(A, b, lower=True, check_finite=False)
        assert_array_almost_equal(sol, [1, 0])