import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_swap(self):
    for p in 'sd':
        f = getattr(fblas, p + 'swap', None)
        if f is None:
            continue
        x, y = ([2, 3, 1], [-2, 3, 7])
        x1, y1 = f(x, y)
        assert_array_almost_equal(x1, y)
        assert_array_almost_equal(y1, x)
    for p in 'cz':
        f = getattr(fblas, p + 'swap', None)
        if f is None:
            continue
        x, y = ([2, 3j, 1], [-2, 3, 7 - 3j])
        x1, y1 = f(x, y)
        assert_array_almost_equal(x1, y)
        assert_array_almost_equal(y1, x)