import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_woodbury(self):
    """
        Random elements in diagonal matrix with blocks in the
        left lower and right upper corners checking the
        implementation of Woodbury algorithm.
        """
    np.random.seed(1234)
    n = 201
    for k in range(3, 32, 2):
        offset = int((k - 1) / 2)
        a = np.diagflat(np.random.random((1, n)))
        for i in range(1, offset + 1):
            a[:-i, i:] += np.diagflat(np.random.random((1, n - i)))
            a[i:, :-i] += np.diagflat(np.random.random((1, n - i)))
        ur = np.random.random((offset, offset))
        a[:offset, -offset:] = ur
        ll = np.random.random((offset, offset))
        a[-offset:, :offset] = ll
        d = np.zeros((k, n))
        for i, j in enumerate(range(offset, -offset - 1, -1)):
            if j < 0:
                d[i, :j] = np.diagonal(a, offset=j)
            else:
                d[i, j:] = np.diagonal(a, offset=j)
        b = np.random.random(n)
        assert_allclose(_woodbury_algorithm(d, ur, ll, b, k), np.linalg.solve(a, b), atol=1e-14)