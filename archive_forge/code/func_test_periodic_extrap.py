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
def test_periodic_extrap(self):
    np.random.seed(1234)
    t = np.sort(np.random.random(8))
    c = np.random.random(4)
    k = 3
    b = BSpline(t, c, k, extrapolate='periodic')
    n = t.size - (k + 1)
    dt = t[-1] - t[0]
    xx = np.linspace(t[k] - dt, t[n] + dt, 50)
    xy = t[k] + (xx - t[k]) % (t[n] - t[k])
    assert_allclose(b(xx), splev(xy, (t, c, k)))
    xx = [-1, 0, 0.5, 1]
    xy = t[k] + (xx - t[k]) % (t[n] - t[k])
    assert_equal(b(xx, extrapolate='periodic'), b(xy, extrapolate=True))