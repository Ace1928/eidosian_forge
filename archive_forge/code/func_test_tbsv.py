import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_tbsv(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 6
        k = 3
        x = rand(n).astype(dtype)
        A = zeros((n, n), dtype=dtype)
        for sup in range(k + 1):
            A[arange(n - sup), arange(sup, n)] = rand(n - sup)
        if ind > 1:
            A[nonzero(A)] += 1j * rand((k + 1) * n - k * (k + 1) // 2).astype(dtype)
        Ab = zeros((k + 1, n), dtype=dtype)
        for row in range(k + 1):
            Ab[-row - 1, row:] = diag(A, k=row)
        func, = get_blas_funcs(('tbsv',), dtype=dtype)
        y1 = func(k=k, a=Ab, x=x)
        y2 = solve(A, x)
        assert_array_almost_equal(y1, y2)
        y1 = func(k=k, a=Ab, x=x, diag=1)
        A[arange(n), arange(n)] = dtype(1)
        y2 = solve(A, x)
        assert_array_almost_equal(y1, y2)
        y1 = func(k=k, a=Ab, x=x, diag=1, trans=1)
        y2 = solve(A.T, x)
        assert_array_almost_equal(y1, y2)
        y1 = func(k=k, a=Ab, x=x, diag=1, trans=2)
        y2 = solve(A.conj().T, x)
        assert_array_almost_equal(y1, y2)