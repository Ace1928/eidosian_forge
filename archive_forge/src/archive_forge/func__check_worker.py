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