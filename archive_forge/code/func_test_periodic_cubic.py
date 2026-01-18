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
def test_periodic_cubic(self):
    b = make_interp_spline(self.xx, self.yy, k=3, bc_type='periodic')
    cub = CubicSpline(self.xx, self.yy, bc_type='periodic')
    assert_allclose(b(self.xx), cub(self.xx), atol=1e-14)
    n = 3
    x = np.sort(np.random.random_sample(n) * 10)
    y = np.random.random_sample(n) * 100
    y[0] = y[-1]
    b = make_interp_spline(x, y, k=3, bc_type='periodic')
    cub = CubicSpline(x, y, bc_type='periodic')
    assert_allclose(b(x), cub(x), atol=1e-14)