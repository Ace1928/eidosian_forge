import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_syr_her(self):
    x = np.arange(1, 5, dtype='d')
    resx = np.triu(x[:, np.newaxis] * x)
    resx_reverse = np.triu(x[::-1, np.newaxis] * x[::-1])
    y = np.linspace(0, 8.5, 17, endpoint=False)
    z = np.arange(1, 9, dtype='d').view('D')
    resz = np.triu(z[:, np.newaxis] * z)
    resz_reverse = np.triu(z[::-1, np.newaxis] * z[::-1])
    rehz = np.triu(z[:, np.newaxis] * z.conj())
    rehz_reverse = np.triu(z[::-1, np.newaxis] * z[::-1].conj())
    w = np.c_[np.zeros(4), z, np.zeros(4)].ravel()
    for p, rtol in zip('sd', [1e-07, 1e-14]):
        f = getattr(fblas, p + 'syr', None)
        if f is None:
            continue
        assert_allclose(f(1.0, x), resx, rtol=rtol)
        assert_allclose(f(1.0, x, lower=True), resx.T, rtol=rtol)
        assert_allclose(f(1.0, y, incx=2, offx=2, n=4), resx, rtol=rtol)
        assert_allclose(f(1.0, y, incx=-2, offx=2, n=4), resx_reverse, rtol=rtol)
        a = np.zeros((4, 4), 'f' if p == 's' else 'd', 'F')
        b = f(1.0, x, a=a, overwrite_a=True)
        assert_allclose(a, resx, rtol=rtol)
        b = f(2.0, x, a=a)
        assert_(a is not b)
        assert_allclose(b, 3 * resx, rtol=rtol)
        assert_raises(Exception, f, 1.0, x, incx=0)
        assert_raises(Exception, f, 1.0, x, offx=5)
        assert_raises(Exception, f, 1.0, x, offx=-2)
        assert_raises(Exception, f, 1.0, x, n=-2)
        assert_raises(Exception, f, 1.0, x, n=5)
        assert_raises(Exception, f, 1.0, x, lower=2)
        assert_raises(Exception, f, 1.0, x, a=np.zeros((2, 2), 'd', 'F'))
    for p, rtol in zip('cz', [1e-07, 1e-14]):
        f = getattr(fblas, p + 'syr', None)
        if f is None:
            continue
        assert_allclose(f(1.0, z), resz, rtol=rtol)
        assert_allclose(f(1.0, z, lower=True), resz.T, rtol=rtol)
        assert_allclose(f(1.0, w, incx=3, offx=1, n=4), resz, rtol=rtol)
        assert_allclose(f(1.0, w, incx=-3, offx=1, n=4), resz_reverse, rtol=rtol)
        a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
        b = f(1.0, z, a=a, overwrite_a=True)
        assert_allclose(a, resz, rtol=rtol)
        b = f(2.0, z, a=a)
        assert_(a is not b)
        assert_allclose(b, 3 * resz, rtol=rtol)
        assert_raises(Exception, f, 1.0, x, incx=0)
        assert_raises(Exception, f, 1.0, x, offx=5)
        assert_raises(Exception, f, 1.0, x, offx=-2)
        assert_raises(Exception, f, 1.0, x, n=-2)
        assert_raises(Exception, f, 1.0, x, n=5)
        assert_raises(Exception, f, 1.0, x, lower=2)
        assert_raises(Exception, f, 1.0, x, a=np.zeros((2, 2), 'd', 'F'))
    for p, rtol in zip('cz', [1e-07, 1e-14]):
        f = getattr(fblas, p + 'her', None)
        if f is None:
            continue
        assert_allclose(f(1.0, z), rehz, rtol=rtol)
        assert_allclose(f(1.0, z, lower=True), rehz.T.conj(), rtol=rtol)
        assert_allclose(f(1.0, w, incx=3, offx=1, n=4), rehz, rtol=rtol)
        assert_allclose(f(1.0, w, incx=-3, offx=1, n=4), rehz_reverse, rtol=rtol)
        a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
        b = f(1.0, z, a=a, overwrite_a=True)
        assert_allclose(a, rehz, rtol=rtol)
        b = f(2.0, z, a=a)
        assert_(a is not b)
        assert_allclose(b, 3 * rehz, rtol=rtol)
        assert_raises(Exception, f, 1.0, x, incx=0)
        assert_raises(Exception, f, 1.0, x, offx=5)
        assert_raises(Exception, f, 1.0, x, offx=-2)
        assert_raises(Exception, f, 1.0, x, n=-2)
        assert_raises(Exception, f, 1.0, x, n=5)
        assert_raises(Exception, f, 1.0, x, lower=2)
        assert_raises(Exception, f, 1.0, x, a=np.zeros((2, 2), 'd', 'F'))