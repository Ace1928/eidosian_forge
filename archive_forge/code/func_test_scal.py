import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_scal(self):
    for p in 'sd':
        f = getattr(fblas, p + 'scal', None)
        if f is None:
            continue
        assert_array_almost_equal(f(2, [3, -4, 5]), [6, -8, 10])
    for p in 'cz':
        f = getattr(fblas, p + 'scal', None)
        if f is None:
            continue
        assert_array_almost_equal(f(3j, [3j, -4, 3 - 4j]), [-9, -12j, 12 + 9j])
    for p in ['cs', 'zd']:
        f = getattr(fblas, p + 'scal', None)
        if f is None:
            continue
        assert_array_almost_equal(f(3, [3j, -4, 3 - 4j]), [9j, -12, 9 - 12j])