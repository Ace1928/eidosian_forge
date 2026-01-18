import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_spr2_hpr2(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 3
        A = rand(n, n).astype(dtype)
        if ind > 1:
            A += rand(n, n) * 1j
        A = A.astype(dtype)
        A = A + A.T if ind < 2 else A + A.conj().T
        c, r = tril_indices(n)
        Ap = A[r, c]
        x = rand(n).astype(dtype)
        y = rand(n).astype(dtype)
        alpha = dtype(2)
        if ind > 1:
            func, = get_blas_funcs(('hpr2',), dtype=dtype)
        else:
            func, = get_blas_funcs(('spr2',), dtype=dtype)
        u = alpha.conj() * x[:, None].dot(y[None, :].conj())
        y2 = A + u + u.conj().T
        y1 = func(n=n, alpha=alpha, x=x, y=y, ap=Ap)
        y1f = zeros((3, 3), dtype=dtype)
        y1f[r, c] = y1
        y1f[[1, 2, 2], [0, 0, 1]] = y1[[1, 3, 4]].conj()
        assert_array_almost_equal(y1f, y2)