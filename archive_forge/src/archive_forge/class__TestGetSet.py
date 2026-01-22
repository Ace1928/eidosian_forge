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
class _TestGetSet:

    def test_getelement(self):

        def check(dtype):
            D = array([[1, 0, 0], [4, 3, 0], [0, 2, 0], [0, 0, 0]], dtype=dtype)
            A = self.spcreator(D)
            M, N = D.shape
            for i in range(-M, M):
                for j in range(-N, N):
                    assert_equal(A[i, j], D[i, j])
            assert_equal(type(A[1, 1]), dtype)
            for ij in [(0, 3), (-1, 3), (4, 0), (4, 3), (4, -1), (1, 2, 3)]:
                assert_raises((IndexError, TypeError), A.__getitem__, ij)
        for dtype in supported_dtypes:
            check(np.dtype(dtype))

    def test_setelement(self):

        def check(dtype):
            A = self.spcreator((3, 4), dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
                A[0, 0] = dtype.type(0)
                A[1, 2] = dtype.type(4.0)
                A[0, 1] = dtype.type(3)
                A[2, 0] = dtype.type(2.0)
                A[0, -1] = dtype.type(8)
                A[-1, -2] = dtype.type(7)
                A[0, 1] = dtype.type(5)
            if dtype != np.bool_:
                assert_array_equal(A.toarray(), [[0, 5, 0, 8], [0, 0, 4, 0], [2, 0, 7, 0]])
            for ij in [(0, 4), (-1, 4), (3, 0), (3, 4), (3, -1)]:
                assert_raises(IndexError, A.__setitem__, ij, 123.0)
            for v in [[1, 2, 3], array([1, 2, 3])]:
                assert_raises(ValueError, A.__setitem__, (0, 0), v)
            if not np.issubdtype(dtype, np.complexfloating) and dtype != np.bool_:
                for v in [3j]:
                    assert_raises(TypeError, A.__setitem__, (0, 0), v)
        for dtype in supported_dtypes:
            check(np.dtype(dtype))

    def test_negative_index_assignment(self):

        def check(dtype):
            A = self.spcreator((3, 10), dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
                A[0, -4] = 1
            assert_equal(A[0, -4], 1)
        for dtype in self.math_dtypes:
            check(np.dtype(dtype))

    def test_scalar_assign_2(self):
        n, m = (5, 10)

        def _test_set(i, j, nitems):
            msg = f'{i!r} ; {j!r} ; {nitems!r}'
            A = self.spcreator((n, m))
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
                A[i, j] = 1
            assert_almost_equal(A.sum(), nitems, err_msg=msg)
            assert_almost_equal(A[i, j], 1, err_msg=msg)
        for i, j in [(2, 3), (-1, 8), (-1, -2), (array(-1), -2), (-1, array(-2)), (array(-1), array(-2))]:
            _test_set(i, j, 1)

    def test_index_scalar_assign(self):
        A = self.spcreator((5, 5))
        B = np.zeros((5, 5))
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            for C in [A, B]:
                C[0, 1] = 1
                C[3, 0] = 4
                C[3, 0] = 9
        assert_array_equal(A.toarray(), B)