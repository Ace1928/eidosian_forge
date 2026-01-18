import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_sbmv_hbmv(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 6
        k = 2
        A = zeros((n, n), dtype=dtype)
        Ab = zeros((k + 1, n), dtype=dtype)
        A[arange(n), arange(n)] = rand(n)
        for ind2 in range(1, k + 1):
            temp = rand(n - ind2)
            A[arange(n - ind2), arange(ind2, n)] = temp
            Ab[-1 - ind2, ind2:] = temp
        A = A.astype(dtype)
        A = A + A.T if ind < 2 else A + A.conj().T
        Ab[-1, :] = diag(A)
        x = rand(n).astype(dtype)
        y = rand(n).astype(dtype)
        alpha, beta = (dtype(1.25), dtype(3))
        if ind > 1:
            func, = get_blas_funcs(('hbmv',), dtype=dtype)
        else:
            func, = get_blas_funcs(('sbmv',), dtype=dtype)
        y1 = func(k=k, alpha=alpha, a=Ab, x=x, y=y, beta=beta)
        y2 = alpha * A.dot(x) + beta * y
        assert_array_almost_equal(y1, y2)