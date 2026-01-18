import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_partial_derivative_method(self):
    x = array([1, 2, 3, 4, 5])
    y = array([1, 2, 3, 4, 5])
    z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
    dx = array([0, 0, 2.0 / 3, 0, 0])
    dy = array([4, -1, 0, -0.25, -4])
    dxdy = array([160, 65, 0, 55, 32]) / 24.0
    lut = RectBivariateSpline(x, y, z)
    assert_array_almost_equal(lut.partial_derivative(1, 0)(x, y, grid=False), dx)
    assert_array_almost_equal(lut.partial_derivative(0, 1)(x, y, grid=False), dy)
    assert_array_almost_equal(lut.partial_derivative(1, 1)(x, y, grid=False), dxdy)