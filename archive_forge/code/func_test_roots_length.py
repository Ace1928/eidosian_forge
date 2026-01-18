import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_roots_length(self):
    x = np.linspace(0, 50 * np.pi, 1000)
    y = np.cos(x)
    spl = UnivariateSpline(x, y, s=0)
    assert_equal(len(spl.roots()), 50)