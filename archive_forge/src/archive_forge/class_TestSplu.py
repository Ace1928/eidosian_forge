import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
class TestSplu:

    def setup_method(self):
        use_solver(useUmfpack=False)
        n = 40
        d = arange(n) + 1
        self.n = n
        self.A = spdiags((d, 2 * d, d[::-1]), (-3, 0, 5), n, n, format='csc')
        random.seed(1234)

    def _smoketest(self, spxlu, check, dtype, idx_dtype):
        if np.issubdtype(dtype, np.complexfloating):
            A = self.A + 1j * self.A.T
        else:
            A = self.A
        A = A.astype(dtype)
        A.indices = A.indices.astype(idx_dtype, copy=False)
        A.indptr = A.indptr.astype(idx_dtype, copy=False)
        lu = spxlu(A)
        rng = random.RandomState(1234)
        for k in [None, 1, 2, self.n, self.n + 2]:
            msg = f'k={k!r}'
            if k is None:
                b = rng.rand(self.n)
            else:
                b = rng.rand(self.n, k)
            if np.issubdtype(dtype, np.complexfloating):
                b = b + 1j * rng.rand(*b.shape)
            b = b.astype(dtype)
            x = lu.solve(b)
            check(A, b, x, msg)
            x = lu.solve(b, 'T')
            check(A.T, b, x, msg)
            x = lu.solve(b, 'H')
            check(A.T.conj(), b, x, msg)

    @sup_sparse_efficiency
    def test_splu_smoketest(self):
        self._internal_test_splu_smoketest()

    def _internal_test_splu_smoketest(self):

        def check(A, b, x, msg=''):
            eps = np.finfo(A.dtype).eps
            r = A @ x
            assert_(abs(r - b).max() < 1000.0 * eps, msg)
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for idx_dtype in [np.int32, np.int64]:
                self._smoketest(splu, check, dtype, idx_dtype)

    @sup_sparse_efficiency
    def test_spilu_smoketest(self):
        self._internal_test_spilu_smoketest()

    def _internal_test_spilu_smoketest(self):
        errors = []

        def check(A, b, x, msg=''):
            r = A @ x
            err = abs(r - b).max()
            assert_(err < 0.01, msg)
            if b.dtype in (np.float64, np.complex128):
                errors.append(err)
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for idx_dtype in [np.int32, np.int64]:
                self._smoketest(spilu, check, dtype, idx_dtype)
        assert_(max(errors) > 1e-05)

    @sup_sparse_efficiency
    def test_spilu_drop_rule(self):
        A = identity(2)
        rules = [b'basic,area'.decode('ascii'), b'basic,area', [b'basic', b'area'.decode('ascii')]]
        for rule in rules:
            assert_(isinstance(spilu(A, drop_rule=rule), SuperLU))

    def test_splu_nnz0(self):
        A = csc_matrix((5, 5), dtype='d')
        assert_raises(RuntimeError, splu, A)

    def test_spilu_nnz0(self):
        A = csc_matrix((5, 5), dtype='d')
        assert_raises(RuntimeError, spilu, A)

    def test_splu_basic(self):
        n = 30
        rng = random.RandomState(12)
        a = rng.rand(n, n)
        a[a < 0.95] = 0
        a[:, 0] = 0
        a_ = csc_matrix(a)
        assert_raises(RuntimeError, splu, a_)
        a += 4 * eye(n)
        a_ = csc_matrix(a)
        lu = splu(a_)
        b = ones(n)
        x = lu.solve(b)
        assert_almost_equal(dot(a, x), b)

    def test_splu_perm(self):
        n = 30
        a = random.random((n, n))
        a[a < 0.95] = 0
        a += 4 * eye(n)
        a_ = csc_matrix(a)
        lu = splu(a_)
        for perm in (lu.perm_r, lu.perm_c):
            assert_(all(perm > -1))
            assert_(all(perm < n))
            assert_equal(len(unique(perm)), len(perm))
        a = a + a.T
        a_ = csc_matrix(a)
        lu = splu(a_)
        assert_array_equal(lu.perm_r, lu.perm_c)

    @pytest.mark.parametrize('splu_fun, rtol', [(splu, 1e-07), (spilu, 0.1)])
    def test_natural_permc(self, splu_fun, rtol):
        np.random.seed(42)
        n = 500
        p = 0.01
        A = scipy.sparse.random(n, n, p)
        x = np.random.rand(n)
        A += (n + 1) * scipy.sparse.identity(n)
        A_ = csc_matrix(A)
        b = A_ @ x
        lu = splu_fun(A_)
        assert_(np.any(lu.perm_c != np.arange(n)))
        lu = splu_fun(A_, permc_spec='NATURAL')
        assert_array_equal(lu.perm_c, np.arange(n))
        x2 = lu.solve(b)
        assert_allclose(x, x2, rtol=rtol)

    @pytest.mark.skipif(not hasattr(sys, 'getrefcount'), reason='no sys.getrefcount')
    def test_lu_refcount(self):
        n = 30
        a = random.random((n, n))
        a[a < 0.95] = 0
        a += 4 * eye(n)
        a_ = csc_matrix(a)
        lu = splu(a_)
        rc = sys.getrefcount(lu)
        for attr in ('perm_r', 'perm_c'):
            perm = getattr(lu, attr)
            assert_equal(sys.getrefcount(lu), rc + 1)
            del perm
            assert_equal(sys.getrefcount(lu), rc)

    def test_bad_inputs(self):
        A = self.A.tocsc()
        assert_raises(ValueError, splu, A[:, :4])
        assert_raises(ValueError, spilu, A[:, :4])
        for lu in [splu(A), spilu(A)]:
            b = random.rand(42)
            B = random.rand(42, 3)
            BB = random.rand(self.n, 3, 9)
            assert_raises(ValueError, lu.solve, b)
            assert_raises(ValueError, lu.solve, B)
            assert_raises(ValueError, lu.solve, BB)
            assert_raises(TypeError, lu.solve, b.astype(np.complex64))
            assert_raises(TypeError, lu.solve, b.astype(np.complex128))

    @sup_sparse_efficiency
    def test_superlu_dlamch_i386_nan(self):
        n = 8
        d = np.arange(n) + 1
        A = spdiags((d, 2 * d, d[::-1]), (-3, 0, 5), n, n)
        A = A.astype(np.float32)
        spilu(A)
        A = A + 1j * A
        B = A.A
        assert_(not np.isnan(B).any())

    @sup_sparse_efficiency
    def test_lu_attr(self):

        def check(dtype, complex_2=False):
            A = self.A.astype(dtype)
            if complex_2:
                A = A + 1j * A.T
            n = A.shape[0]
            lu = splu(A)
            Pc = np.zeros((n, n))
            Pc[np.arange(n), lu.perm_c] = 1
            Pr = np.zeros((n, n))
            Pr[lu.perm_r, np.arange(n)] = 1
            Ad = A.toarray()
            lhs = Pr.dot(Ad).dot(Pc)
            rhs = (lu.L @ lu.U).toarray()
            eps = np.finfo(dtype).eps
            assert_allclose(lhs, rhs, atol=100 * eps)
        check(np.float32)
        check(np.float64)
        check(np.complex64)
        check(np.complex128)
        check(np.complex64, True)
        check(np.complex128, True)

    @pytest.mark.slow
    @sup_sparse_efficiency
    def test_threads_parallel(self):
        oks = []

        def worker():
            try:
                self.test_splu_basic()
                self._internal_test_splu_smoketest()
                self._internal_test_spilu_smoketest()
                oks.append(True)
            except Exception:
                pass
        threads = [threading.Thread(target=worker) for k in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert_equal(len(oks), 20)