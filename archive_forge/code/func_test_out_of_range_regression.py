import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_out_of_range_regression(self):
    x = np.arange(5, dtype=float)
    y = x ** 3
    xp = linspace(-8, 13, 100)
    xp_zeros = xp.copy()
    xp_zeros[np.logical_or(xp_zeros < 0.0, xp_zeros > 4.0)] = 0
    xp_clip = xp.copy()
    xp_clip[xp_clip < x[0]] = x[0]
    xp_clip[xp_clip > x[-1]] = x[-1]
    for cls in [UnivariateSpline, InterpolatedUnivariateSpline]:
        spl = cls(x=x, y=y)
        for ext in [0, 'extrapolate']:
            assert_allclose(spl(xp, ext=ext), xp ** 3, atol=1e-16)
            assert_allclose(cls(x, y, ext=ext)(xp), xp ** 3, atol=1e-16)
        for ext in [1, 'zeros']:
            assert_allclose(spl(xp, ext=ext), xp_zeros ** 3, atol=1e-16)
            assert_allclose(cls(x, y, ext=ext)(xp), xp_zeros ** 3, atol=1e-16)
        for ext in [2, 'raise']:
            assert_raises(ValueError, spl, xp, **dict(ext=ext))
        for ext in [3, 'const']:
            assert_allclose(spl(xp, ext=ext), xp_clip ** 3, atol=1e-16)
            assert_allclose(cls(x, y, ext=ext)(xp), xp_clip ** 3, atol=1e-16)
    t = spl.get_knots()[3:4]
    spl = LSQUnivariateSpline(x, y, t)
    assert_allclose(spl(xp, ext=0), xp ** 3, atol=1e-16)
    assert_allclose(spl(xp, ext=1), xp_zeros ** 3, atol=1e-16)
    assert_raises(ValueError, spl, xp, **dict(ext=2))
    assert_allclose(spl(xp, ext=3), xp_clip ** 3, atol=1e-16)
    for ext in [-1, 'unknown']:
        spl = UnivariateSpline(x, y)
        assert_raises(ValueError, spl, xp, **dict(ext=ext))
        assert_raises(ValueError, UnivariateSpline, **dict(x=x, y=y, ext=ext))