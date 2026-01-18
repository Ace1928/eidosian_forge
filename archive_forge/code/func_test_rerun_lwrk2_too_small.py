import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_rerun_lwrk2_too_small(self):
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-2, 2, 80)
    z = x + y
    xi = np.linspace(-1, 1, 100)
    yi = np.linspace(-2, 2, 100)
    tck = bisplrep(x, y, z)
    res1 = bisplev(xi, yi, tck)
    interp_ = SmoothBivariateSpline(x, y, z)
    res2 = interp_(xi, yi)
    assert_almost_equal(res1, res2)