import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_preserve_shape(self):
    x = [1, 2, 3]
    y = [0, 2, 4]
    lut = UnivariateSpline(x, y, k=1)
    arg = 2
    assert_equal(shape(arg), shape(lut(arg)))
    assert_equal(shape(arg), shape(lut(arg, nu=1)))
    arg = [1.5, 2, 2.5]
    assert_equal(shape(arg), shape(lut(arg)))
    assert_equal(shape(arg), shape(lut(arg, nu=1)))