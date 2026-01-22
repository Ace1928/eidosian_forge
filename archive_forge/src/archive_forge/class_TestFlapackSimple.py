import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
class TestFlapackSimple:

    def test_gebal(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a1 = [[1, 0, 0, 0.0003], [4, 0, 0, 0.002], [7, 1, 0, 0], [0, 1, 0, 0]]
        for p in 'sdzc':
            f = getattr(flapack, p + 'gebal', None)
            if f is None:
                continue
            ba, lo, hi, pivscale, info = f(a)
            assert_(not info, repr(info))
            assert_array_almost_equal(ba, a)
            assert_equal((lo, hi), (0, len(a[0]) - 1))
            assert_array_almost_equal(pivscale, np.ones(len(a)))
            ba, lo, hi, pivscale, info = f(a1, permute=1, scale=1)
            assert_(not info, repr(info))

    def test_gehrd(self):
        a = [[-149, -50, -154], [537, 180, 546], [-27, -9, -25]]
        for p in 'd':
            f = getattr(flapack, p + 'gehrd', None)
            if f is None:
                continue
            ht, tau, info = f(a)
            assert_(not info, repr(info))

    def test_trsyl(self):
        a = np.array([[1, 2], [0, 4]])
        b = np.array([[5, 6], [0, 8]])
        c = np.array([[9, 10], [11, 12]])
        trans = 'T'
        for dtype in 'fdFD':
            a1, b1, c1 = (a.astype(dtype), b.astype(dtype), c.astype(dtype))
            trsyl, = get_lapack_funcs(('trsyl',), (a1,))
            if dtype.isupper():
                a1[0] += 1j
                trans = 'C'
            x, scale, info = trsyl(a1, b1, c1)
            assert_array_almost_equal(np.dot(a1, x) + np.dot(x, b1), scale * c1)
            x, scale, info = trsyl(a1, b1, c1, trana=trans, tranb=trans)
            assert_array_almost_equal(np.dot(a1.conjugate().T, x) + np.dot(x, b1.conjugate().T), scale * c1, decimal=4)
            x, scale, info = trsyl(a1, b1, c1, isgn=-1)
            assert_array_almost_equal(np.dot(a1, x) - np.dot(x, b1), scale * c1, decimal=4)

    def test_lange(self):
        a = np.array([[-149, -50, -154], [537, 180, 546], [-27, -9, -25]])
        for dtype in 'fdFD':
            for norm_str in 'Mm1OoIiFfEe':
                a1 = a.astype(dtype)
                if dtype.isupper():
                    a1[0, 0] += 1j
                lange, = get_lapack_funcs(('lange',), (a1,))
                value = lange(norm_str, a1)
                if norm_str in 'FfEe':
                    if dtype in 'Ff':
                        decimal = 3
                    else:
                        decimal = 7
                    ref = np.sqrt(np.sum(np.square(np.abs(a1))))
                    assert_almost_equal(value, ref, decimal)
                else:
                    if norm_str in 'Mm':
                        ref = np.max(np.abs(a1))
                    elif norm_str in '1Oo':
                        ref = np.max(np.sum(np.abs(a1), axis=0))
                    elif norm_str in 'Ii':
                        ref = np.max(np.sum(np.abs(a1), axis=1))
                    assert_equal(value, ref)