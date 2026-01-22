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
class TestLinalgCholesky(TestLinalgBase):
    """
    Tests for np.linalg.cholesky.
    """

    def sample_matrix(self, m, dtype, order):
        np.random.seed(0)
        A = np.random.rand(m, m)
        q, _ = np.linalg.qr(A)
        L = np.arange(1, m + 1)
        Q = np.dot(np.dot(q.T, np.diag(L)), q)
        Q = np.array(Q, dtype=dtype, order=order)
        return Q

    def assert_not_pd(self, cfunc, args):
        msg = 'Matrix is not positive definite.'
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    @needs_lapack
    def test_linalg_cholesky(self):
        """
        Test np.linalg.cholesky
        """
        n = 10
        cfunc = jit(nopython=True)(cholesky_matrix)

        def check(a):
            expected = cholesky_matrix(a)
            got = cfunc(a)
            use_reconstruction = False
            self.assert_contig_sanity(got, 'C')
            try:
                np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
            except AssertionError:
                use_reconstruction = True
            if use_reconstruction:
                rec = np.dot(got, np.conj(got.T))
                resolution = 5 * np.finfo(a.dtype).resolution
                np.testing.assert_allclose(a, rec, rtol=resolution, atol=resolution)
            with self.assertNoNRTLeak():
                cfunc(a)
        for dtype, order in product(self.dtypes, 'FC'):
            a = self.sample_matrix(n, dtype, order)
            check(a)
        check(np.empty((0, 0)))
        rn = 'cholesky'
        self.assert_non_square(cfunc, (np.ones((2, 3), dtype=np.float64),))
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64),))
        self.assert_not_pd(cfunc, (np.ones(4, dtype=np.float64).reshape(2, 2),))