import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
class _TestFancyIndexingAssign:

    def test_bad_index_assign(self):
        A = self.spcreator(np.zeros([5, 5]))
        assert_raises((IndexError, ValueError, TypeError), A.__setitem__, 'foo', 2)
        assert_raises((IndexError, ValueError, TypeError), A.__setitem__, (2, 'foo'), 5)

    def test_fancy_indexing_set(self):
        n, m = (5, 10)

        def _test_set_slice(i, j):
            A = self.spcreator((n, m))
            B = asmatrix(np.zeros((n, m)))
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
                B[i, j] = 1
                with check_remains_sorted(A):
                    A[i, j] = 1
            assert_array_almost_equal(A.toarray(), B)
        for i, j in [((2, 3, 4), slice(None, 10, 4)), (np.arange(3), slice(5, -2)), (slice(2, 5), slice(5, -2))]:
            _test_set_slice(i, j)
        for i, j in [(np.arange(3), np.arange(3)), ((0, 3, 4), (1, 2, 4))]:
            _test_set_slice(i, j)

    def test_fancy_assignment_dtypes(self):

        def check(dtype):
            A = self.spcreator((5, 5), dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
                A[[0, 1], [0, 1]] = dtype.type(1)
                assert_equal(A.sum(), dtype.type(1) * 2)
                A[0:2, 0:2] = dtype.type(1.0)
                assert_equal(A.sum(), dtype.type(1) * 4)
                A[2, 2] = dtype.type(1.0)
                assert_equal(A.sum(), dtype.type(1) * 4 + dtype.type(1))
        for dtype in supported_dtypes:
            check(np.dtype(dtype))

    def test_sequence_assignment(self):
        A = self.spcreator((4, 3))
        B = self.spcreator(eye(3, 4))
        i0 = [0, 1, 2]
        i1 = (0, 1, 2)
        i2 = array(i0)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            with check_remains_sorted(A):
                A[0, i0] = B[i0, 0].T
                A[1, i1] = B[i1, 1].T
                A[2, i2] = B[i2, 2].T
            assert_array_equal(A.toarray(), B.T.toarray())
            A = self.spcreator((2, 3))
            with check_remains_sorted(A):
                A[1, 1:3] = [10, 20]
            assert_array_equal(A.toarray(), [[0, 0, 0], [0, 10, 20]])
            A = self.spcreator((3, 2))
            with check_remains_sorted(A):
                A[1:3, 1] = [[10], [20]]
            assert_array_equal(A.toarray(), [[0, 0], [0, 10], [0, 20]])
            A = self.spcreator((3, 3))
            B = asmatrix(np.zeros((3, 3)))
            with check_remains_sorted(A):
                for C in [A, B]:
                    C[[0, 1, 2], [0, 1, 2]] = [4, 5, 6]
            assert_array_equal(A.toarray(), B)
            A = self.spcreator((4, 3))
            with check_remains_sorted(A):
                A[(1, 2, 3), (0, 1, 2)] = [1, 2, 3]
            assert_almost_equal(A.sum(), 6)
            B = asmatrix(np.zeros((4, 3)))
            B[(1, 2, 3), (0, 1, 2)] = [1, 2, 3]
            assert_array_equal(A.toarray(), B)

    def test_fancy_assign_empty(self):
        B = asmatrix(arange(50).reshape(5, 10))
        B[1, :] = 0
        B[:, 2] = 0
        B[3, 6] = 0
        A = self.spcreator(B)
        K = np.array([False, False, False, False, False])
        A[K] = 42
        assert_equal(toarray(A), B)
        K = np.array([], dtype=int)
        A[K] = 42
        assert_equal(toarray(A), B)
        A[K, K] = 42
        assert_equal(toarray(A), B)
        J = np.array([0, 1, 2, 3, 4], dtype=int)[:, None]
        A[K, J] = 42
        assert_equal(toarray(A), B)
        A[J, K] = 42
        assert_equal(toarray(A), B)