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
class TestCumulative_trapezoid:

    def test_1d(self):
        x = np.linspace(-2, 2, num=5)
        y = x
        y_int = cumulative_trapezoid(y, x, initial=0)
        y_expected = [0.0, -1.5, -2.0, -1.5, 0.0]
        assert_allclose(y_int, y_expected)
        y_int = cumulative_trapezoid(y, x, initial=None)
        assert_allclose(y_int, y_expected[1:])

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

    def test_y_nd_x_1d(self):
        y = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        x = np.arange(4) ** 2
        ys_expected = (np.array([[[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]], [[40.0, 44.0, 48.0, 52.0], [56.0, 60.0, 64.0, 68.0]]]), np.array([[[2.0, 3.0, 4.0, 5.0]], [[10.0, 11.0, 12.0, 13.0]], [[18.0, 19.0, 20.0, 21.0]]]), np.array([[[0.5, 5.0, 17.5], [4.5, 21.0, 53.5]], [[8.5, 37.0, 89.5], [12.5, 53.0, 125.5]], [[16.5, 69.0, 161.5], [20.5, 85.0, 197.5]]]))
        for axis, y_expected in zip([0, 1, 2], ys_expected):
            y_int = cumulative_trapezoid(y, x=x[:y.shape[axis]], axis=axis, initial=None)
            assert_allclose(y_int, y_expected)

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

    @pytest.mark.parametrize('initial', [1, 0.5])
    def test_initial_warning(self, initial):
        """If initial is not None or 0, a ValueError is raised."""
        y = np.linspace(0, 10, num=10)
        with pytest.deprecated_call(match='`initial`'):
            res = cumulative_trapezoid(y, initial=initial)
        assert_allclose(res, [initial, *np.cumsum(y[1:] + y[:-1]) / 2])

    def test_zero_len_y(self):
        with pytest.raises(ValueError, match='At least one point is required'):
            cumulative_trapezoid(y=[])

    def test_cumtrapz(self):
        x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        y = x
        with pytest.deprecated_call(match='cumulative_trapezoid'):
            assert_allclose(cumulative_trapezoid(y, x, dx=0.5, axis=0, initial=0), cumtrapz(y, x, dx=0.5, axis=0, initial=0), rtol=1e-14)