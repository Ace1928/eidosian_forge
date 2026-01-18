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
def test_weighted_smoothing_spline(self):
    np.random.seed(1234)
    n = 100
    x = np.sort(np.random.random_sample(n) * 4 - 2)
    y = x ** 2 * np.sin(4 * x) + x ** 3 + np.random.normal(0.0, 1.5, n)
    spl = make_smoothing_spline(x, y)
    for ind in np.random.choice(range(100), size=10):
        w = np.ones(n)
        w[ind] = 30.0
        spl_w = make_smoothing_spline(x, y, w)
        orig = abs(spl(x[ind]) - y[ind])
        weighted = abs(spl_w(x[ind]) - y[ind])
        if orig < weighted:
            raise ValueError(f'Spline with weights should be closer to the points than the original one: {orig:.4} < {weighted:.4}')