import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_spmv_hpmv(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES + COMPLEX_DTYPES):
        n = 3
        A = rand(n, n).astype(dtype)
        if ind > 1:
            A += rand(n, n) * 1j
        A = A.astype(dtype)
        A = A + A.T if ind < 4 else A + A.conj().T
        c, r = tril_indices(n)
        Ap = A[r, c]
        x = rand(n).astype(dtype)
        y = rand(n).astype(dtype)
        xlong = arange(2 * n).astype(dtype)
        ylong = ones(2 * n).astype(dtype)
        alpha, beta = (dtype(1.25), dtype(2))
        if ind > 3:
            func, = get_blas_funcs(('hpmv',), dtype=dtype)
        else:
            func, = get_blas_funcs(('spmv',), dtype=dtype)
        y1 = func(n=n, alpha=alpha, ap=Ap, x=x, y=y, beta=beta)
        y2 = alpha * A.dot(x) + beta * y
        assert_array_almost_equal(y1, y2)
        y1 = func(n=n - 1, alpha=alpha, beta=beta, x=xlong, y=ylong, ap=Ap, incx=2, incy=2, offx=n, offy=n)
        y2 = (alpha * A[:-1, :-1]).dot(xlong[3::2]) + beta * ylong[3::2]
        assert_array_almost_equal(y1[3::2], y2)
        assert_almost_equal(y1[4], ylong[4])