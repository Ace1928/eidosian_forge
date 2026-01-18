import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
def test_y_nd_x_nd(self):
    x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
    y = x
    y_int = cumulative_trapezoid(y, x, initial=0)
    y_expected = np.array([[[0.0, 0.5, 2.0, 4.5], [0.0, 4.5, 10.0, 16.5]], [[0.0, 8.5, 18.0, 28.5], [0.0, 12.5, 26.0, 40.5]], [[0.0, 16.5, 34.0, 52.5], [0.0, 20.5, 42.0, 64.5]]])
    assert_allclose(y_int, y_expected)
    shapes = [(2, 2, 4), (3, 1, 4), (3, 2, 3)]
    for axis, shape in zip([0, 1, 2], shapes):
        y_int = cumulative_trapezoid(y, x, initial=0, axis=axis)
        assert_equal(y_int.shape, (3, 2, 4))
        y_int = cumulative_trapezoid(y, x, initial=None, axis=axis)
        assert_equal(y_int.shape, shape)