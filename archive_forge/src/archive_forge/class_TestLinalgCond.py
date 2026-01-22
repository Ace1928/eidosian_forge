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
class TestLinalgCond(TestLinalgBase):
    """
    Tests for np.linalg.cond.
    """

    @needs_lapack
    def test_linalg_cond(self):
        """
        Test np.linalg.cond
        """
        cfunc = jit(nopython=True)(cond_matrix)

        def check(a, **kwargs):
            expected = cond_matrix(a, **kwargs)
            got = cfunc(a, **kwargs)
            self.assertTrue(not np.iscomplexobj(got))
            resolution = 5 * np.finfo(a.dtype).resolution
            np.testing.assert_allclose(got, expected, rtol=resolution)
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)
        ps = [None, np.inf, -np.inf, 1, -1, 2, -2]
        sizes = [(3, 3), (7, 7)]
        for size, dtype, order, p in product(sizes, self.dtypes, 'FC', ps):
            a = self.specific_sample_matrix(size, dtype, order)
            check(a, p=p)
        sizes = [(7, 1), (11, 5), (5, 11), (1, 7)]
        for size, dtype, order in product(sizes, self.dtypes, 'FC'):
            a = self.specific_sample_matrix(size, dtype, order)
            check(a)
        for sz in [(0, 1), (1, 0), (0, 0)]:
            self.assert_raise_on_empty(cfunc, (np.empty(sz),))
        x = np.array([[1, 0], [0, 0]], dtype=np.float64)
        check(x)
        check(x, p=2)
        x = np.array([[0, 0], [0, 0]], dtype=np.float64)
        check(x, p=-2)
        with warnings.catch_warnings():
            a = np.array([[1e+308, 0], [0, 0.1]], dtype=np.float64)
            warnings.simplefilter('ignore', RuntimeWarning)
            check(a)
        rn = 'cond'
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64),))
        self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64),))
        self.assert_invalid_norm_kind(cfunc, (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), 6))