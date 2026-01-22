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
class TestBSR(sparse_test_class(getset=False, slicing=False, slicing_assign=False, fancy_indexing=False, fancy_assign=False, nnz_axis=False)):
    spcreator = bsr_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]

    def test_constructor1(self):
        indptr = array([0, 2, 2, 4])
        indices = array([0, 2, 2, 3])
        data = zeros((4, 2, 3))
        data[0] = array([[0, 1, 2], [3, 0, 5]])
        data[1] = array([[0, 2, 4], [6, 0, 10]])
        data[2] = array([[0, 4, 8], [12, 0, 20]])
        data[3] = array([[0, 5, 10], [15, 0, 25]])
        A = kron([[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 4, 5]], [[0, 1, 2], [3, 0, 5]])
        Asp = bsr_matrix((data, indices, indptr), shape=(6, 12))
        assert_equal(Asp.toarray(), A)
        Asp = bsr_matrix((data, indices, indptr))
        assert_equal(Asp.toarray(), A)

    def test_constructor2(self):
        for shape in [(1, 1), (5, 1), (1, 10), (10, 4), (3, 7), (2, 1)]:
            A = zeros(shape)
            assert_equal(bsr_matrix(A).toarray(), A)
        A = zeros((4, 6))
        assert_equal(bsr_matrix(A, blocksize=(2, 2)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 3)).toarray(), A)
        A = kron([[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 4, 5]], [[0, 1, 2], [3, 0, 5]])
        assert_equal(bsr_matrix(A).toarray(), A)
        assert_equal(bsr_matrix(A, shape=(6, 12)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(1, 1)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 3)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 6)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(2, 12)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(3, 12)).toarray(), A)
        assert_equal(bsr_matrix(A, blocksize=(6, 12)).toarray(), A)
        A = kron([[1, 0, 2, 0], [0, 1, 0, 0], [0, 0, 0, 0]], [[0, 1, 2], [3, 0, 5]])
        assert_equal(bsr_matrix(A, blocksize=(2, 3)).toarray(), A)

    def test_constructor3(self):
        arg = ([1, 2, 3], ([0, 1, 1], [0, 0, 1]))
        A = array([[1, 0], [2, 3]])
        assert_equal(bsr_matrix(arg, blocksize=(2, 2)).toarray(), A)

    def test_constructor4(self):
        n = 8
        data = np.ones((n, n, 1), dtype=np.int8)
        indptr = np.array([0, n], dtype=np.int32)
        indices = np.arange(n, dtype=np.int32)
        bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)

    def test_constructor5(self):
        n = 8
        data_1dim = np.ones(n)
        data = np.ones((n, n, n))
        indptr = np.array([0, n])
        indices = np.arange(n)
        with assert_raises(ValueError):
            bsr_matrix((data_1dim, indices, indptr))
        with assert_raises(ValueError):
            bsr_matrix((data, indices, indptr), blocksize=(1, 1, 1))
        with assert_raises(ValueError):
            bsr_matrix((data, indices, indptr), blocksize=(1, 1))

    def test_default_dtype(self):
        values = [[[1], [1]], [[1], [1]]]
        indptr = np.array([0, 2], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)
        b = bsr_matrix((values, indices, indptr), blocksize=(2, 1))
        assert b.dtype == np.array(values).dtype

    def test_bsr_tocsr(self):
        indptr = array([0, 2, 2, 4])
        indices = array([0, 2, 2, 3])
        data = zeros((4, 2, 3))
        data[0] = array([[0, 1, 2], [3, 0, 5]])
        data[1] = array([[0, 2, 4], [6, 0, 10]])
        data[2] = array([[0, 4, 8], [12, 0, 20]])
        data[3] = array([[0, 5, 10], [15, 0, 25]])
        A = kron([[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 4, 5]], [[0, 1, 2], [3, 0, 5]])
        Absr = bsr_matrix((data, indices, indptr), shape=(6, 12))
        Acsr = Absr.tocsr()
        Acsr_via_coo = Absr.tocoo().tocsr()
        assert_equal(Acsr.toarray(), A)
        assert_equal(Acsr.toarray(), Acsr_via_coo.toarray())

    def test_eliminate_zeros(self):
        data = kron([1, 0, 0, 0, 2, 0, 3, 0], [[1, 1], [1, 1]]).T
        data = data.reshape(-1, 2, 2)
        indices = array([1, 2, 3, 4, 5, 6, 7, 8])
        indptr = array([0, 3, 8])
        asp = bsr_matrix((data, indices, indptr), shape=(4, 20))
        bsp = asp.copy()
        asp.eliminate_zeros()
        assert_array_equal(asp.nnz, 3 * 4)
        assert_array_equal(asp.toarray(), bsp.toarray())

    def test_eliminate_zeros_all_zero(self):
        np.random.seed(0)
        m = bsr_matrix(np.random.random((12, 12)), blocksize=(2, 3))
        m.data[m.data <= 0.9] = 0
        m.eliminate_zeros()
        assert_equal(m.nnz, 66)
        assert_array_equal(m.data.shape, (11, 2, 3))
        m.data[m.data <= 1.0] = 0
        m.eliminate_zeros()
        assert_equal(m.nnz, 0)
        assert_array_equal(m.data.shape, (0, 2, 3))
        assert_array_equal(m.toarray(), np.zeros((12, 12)))
        m.eliminate_zeros()
        assert_equal(m.nnz, 0)
        assert_array_equal(m.data.shape, (0, 2, 3))
        assert_array_equal(m.toarray(), np.zeros((12, 12)))

    def test_bsr_matvec(self):
        A = bsr_matrix(arange(2 * 3 * 4 * 5).reshape(2 * 4, 3 * 5), blocksize=(4, 5))
        x = arange(A.shape[1]).reshape(-1, 1)
        assert_equal(A * x, A.toarray() @ x)

    def test_bsr_matvecs(self):
        A = bsr_matrix(arange(2 * 3 * 4 * 5).reshape(2 * 4, 3 * 5), blocksize=(4, 5))
        x = arange(A.shape[1] * 6).reshape(-1, 6)
        assert_equal(A * x, A.toarray() @ x)

    @pytest.mark.xfail(run=False, reason='BSR does not have a __getitem__')
    def test_iterator(self):
        pass

    @pytest.mark.xfail(run=False, reason='BSR does not have a __setitem__')
    def test_setdiag(self):
        pass

    def test_resize_blocked(self):
        D = np.array([[1, 0, 3, 4], [2, 0, 0, 0], [3, 0, 0, 0]])
        S = self.spcreator(D, blocksize=(1, 2))
        assert_(S.resize((3, 2)) is None)
        assert_array_equal(S.toarray(), [[1, 0], [2, 0], [3, 0]])
        S.resize((2, 2))
        assert_array_equal(S.toarray(), [[1, 0], [2, 0]])
        S.resize((3, 2))
        assert_array_equal(S.toarray(), [[1, 0], [2, 0], [0, 0]])
        S.resize((3, 4))
        assert_array_equal(S.toarray(), [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
        assert_raises(ValueError, S.resize, (2, 3))

    @pytest.mark.xfail(run=False, reason='BSR does not have a __setitem__')
    def test_setdiag_comprehensive(self):
        pass

    @pytest.mark.skipif(IS_COLAB, reason='exceeds memory limit')
    def test_scalar_idx_dtype(self):
        indptr = np.zeros(2, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        vals = np.zeros((0, 1, 1))
        a = bsr_matrix((vals, indices, indptr), shape=(1, 2 ** 31 - 1))
        b = bsr_matrix((vals, indices, indptr), shape=(1, 2 ** 31))
        c = bsr_matrix((1, 2 ** 31 - 1))
        d = bsr_matrix((1, 2 ** 31))
        assert_equal(a.indptr.dtype, np.int32)
        assert_equal(b.indptr.dtype, np.int64)
        assert_equal(c.indptr.dtype, np.int32)
        assert_equal(d.indptr.dtype, np.int64)
        try:
            vals2 = np.zeros((0, 1, 2 ** 31 - 1))
            vals3 = np.zeros((0, 1, 2 ** 31))
            e = bsr_matrix((vals2, indices, indptr), shape=(1, 2 ** 31 - 1))
            f = bsr_matrix((vals3, indices, indptr), shape=(1, 2 ** 31))
            assert_equal(e.indptr.dtype, np.int32)
            assert_equal(f.indptr.dtype, np.int64)
        except (MemoryError, ValueError):
            e = 0
            f = 0
        for x in [a, b, c, d, e, f]:
            x + x