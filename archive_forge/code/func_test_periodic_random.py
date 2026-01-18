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
@pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
def test_periodic_random(self, k):
    n = 5
    np.random.seed(1234)
    x = np.sort(np.random.random_sample(n) * 10)
    y = np.random.random_sample(n) * 100
    y[0] = y[-1]
    b = make_interp_spline(x, y, k=k, bc_type='periodic')
    assert_allclose(b(x), y, atol=1e-14)