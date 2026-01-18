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
def test_periodic_axis(self):
    n = self.xx.shape[0]
    np.random.seed(1234)
    x = np.random.random_sample(n) * 2 * np.pi
    x = np.sort(x)
    x[0] = 0.0
    x[-1] = 2 * np.pi
    y = np.zeros((2, n))
    y[0] = np.sin(x)
    y[1] = np.cos(x)
    b = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
    for i in range(n):
        assert_allclose(b(x[i]), y[:, i], atol=1e-14)
    assert_allclose(b(x[0]), b(x[-1]), atol=1e-14)