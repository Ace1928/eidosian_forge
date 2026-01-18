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
def test_x_none(self):
    y = np.linspace(-2, 2, num=5)
    y_int = cumulative_trapezoid(y)
    y_expected = [-1.5, -2.0, -1.5, 0.0]
    assert_allclose(y_int, y_expected)
    y_int = cumulative_trapezoid(y, initial=0)
    y_expected = [0, -1.5, -2.0, -1.5, 0.0]
    assert_allclose(y_int, y_expected)
    y_int = cumulative_trapezoid(y, dx=3)
    y_expected = [-4.5, -6.0, -4.5, 0.0]
    assert_allclose(y_int, y_expected)
    y_int = cumulative_trapezoid(y, dx=3, initial=0)
    y_expected = [0, -4.5, -6.0, -4.5, 0.0]
    assert_allclose(y_int, y_expected)