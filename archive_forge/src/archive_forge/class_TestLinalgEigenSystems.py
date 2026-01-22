import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
class TestLinalgEigenSystems(TestLinalgBase):
    """
    Tests for np.linalg.eig/eigvals.
    """

    def sample_matrix(self, m, dtype, order):
        v = self.sample_vector(m, dtype)
        Q = np.diag(v)
        idx = np.nonzero(np.eye(Q.shape[0], Q.shape[1], 1))
        Q[idx] = v[1:]
        idx = np.nonzero(np.eye(Q.shape[0], Q.shape[1], -1))
        Q[idx] = v[:-1]
        Q = np.array(Q, dtype=dtype, order=order)
        return Q

    def assert_no_domain_change(self, name, cfunc, args):
        msg = name + '() argument must not cause a domain change.'
        self.assert_error(cfunc, args, msg)

    def _check_worker(self, cfunc, name, expected_res_len, check_for_domain_change):

        def check(*args):
            expected = cfunc.py_func(*args)
            got = cfunc(*args)
            a = args[0]
            self.assertEqual(len(expected), len(got))
            res_is_tuple = False
            if isinstance(got, tuple):
                res_is_tuple = True
                self.assertEqual(len(got), expected_res_len)
            else:
                self.assertEqual(got.ndim, expected_res_len)
            self.assert_contig_sanity(got, 'F')
            use_reconstruction = False
            for k in range(len(expected)):
                try:
                    np.testing.assert_array_almost_equal_nulp(got[k], expected[k], nulp=10)
                except AssertionError:
                    use_reconstruction = True
            resolution = 5 * np.finfo(a.dtype).resolution
            if use_reconstruction:
                if res_is_tuple:
                    w, v = got
                    if name[-1] == 'h':
                        idxl = np.nonzero(np.eye(a.shape[0], a.shape[1], -1))
                        idxu = np.nonzero(np.eye(a.shape[0], a.shape[1], 1))
                        cfunc(*args)
                        a[idxu] = np.conj(a[idxl])
                        a[np.diag_indices(a.shape[0])] = np.real(np.diag(a))
                    lhs = np.dot(a, v)
                    rhs = np.dot(v, np.diag(w))
                    np.testing.assert_allclose(lhs.real, rhs.real, rtol=resolution, atol=resolution)
                    if np.iscomplexobj(v):
                        np.testing.assert_allclose(lhs.imag, rhs.imag, rtol=resolution, atol=resolution)
                else:
                    np.testing.assert_allclose(np.sort(expected), np.sort(got), rtol=resolution, atol=resolution)
            with self.assertNoNRTLeak():
                cfunc(*args)
        return check

    def checker_for_linalg_eig(self, name, func, expected_res_len, check_for_domain_change=None):
        """
        Test np.linalg.eig
        """
        n = 10
        cfunc = jit(nopython=True)(func)
        check = self._check_worker(cfunc, name, expected_res_len, check_for_domain_change)
        for dtype, order in product(self.dtypes, 'FC'):
            a = self.sample_matrix(n, dtype, order)
            check(a)
        for ty in [np.float32, np.complex64]:
            check(np.empty((0, 0), dtype=ty))
            self.assert_non_square(cfunc, (np.ones((2, 3), dtype=ty),))
            self.assert_wrong_dtype(name, cfunc, (np.ones((2, 2), dtype=np.int32),))
            self.assert_wrong_dimensions(name, cfunc, (np.ones(10, dtype=ty),))
            self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=ty),))
        if check_for_domain_change:
            A = np.array([[1, -2], [2, 1]])
            check(A.astype(np.complex128))
            l, _ = func(A)
            self.assertTrue(np.any(l.imag))
            for ty in [np.float32, np.float64]:
                self.assert_no_domain_change(name, cfunc, (A.astype(ty),))

    @needs_lapack
    def test_linalg_eig(self):
        self.checker_for_linalg_eig('eig', eig_matrix, 2, True)

    @needs_lapack
    def test_linalg_eigvals(self):
        self.checker_for_linalg_eig('eigvals', eigvals_matrix, 1, True)

    @needs_lapack
    def test_linalg_eigh(self):
        self.checker_for_linalg_eig('eigh', eigh_matrix, 2, False)

    @needs_lapack
    def test_linalg_eigvalsh(self):
        self.checker_for_linalg_eig('eigvalsh', eigvalsh_matrix, 1, False)

    @needs_lapack
    def test_no_input_mutation(self):
        for c in (('eig', 2, True), ('eigvals', 1, True), ('eigh', 2, False), ('eigvalsh', 1, False)):
            m, nout, domain_change = c
            meth = getattr(np.linalg, m)

            @jit(nopython=True)
            def func(X, test):
                if test:
                    X = X[1:2, :]
                return meth(X)
            check = self._check_worker(func, m, nout, domain_change)
            for dtype in (np.float64, np.complex128):
                with self.subTest(meth=meth, dtype=dtype):
                    X = np.array([[10.0, 1, 0, 1], [1, 9, 0, 0], [0, 0, 8, 0], [1, 0, 0, 7]], order='F', dtype=dtype)
                    X_orig = np.copy(X)
                    expected = func.py_func(X, False)
                    np.testing.assert_allclose(X, X_orig)
                    got = func(X, False)
                    np.testing.assert_allclose(X, X_orig)
                    check(X, False)