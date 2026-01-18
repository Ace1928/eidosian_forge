import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_partial_derivative_method_grid(self):
    x = array([1, 2, 3, 4, 5])
    y = array([1, 2, 3, 4, 5])
    z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
    dx = array([[0, 0, -20, 0, 0], [0, 0, 13, 0, 0], [0, 0, 4, 0, 0], [0, 0, -11, 0, 0], [0, 0, 4, 0, 0]]) / 6.0
    dy = array([[4, -1, 0, 1, -4], [4, -1, 0, 1, -4], [0, 1.5, 0, -1.5, 0], [2, 0.25, 0, -0.25, -2], [4, -1, 0, 1, -4]])
    dxdy = array([[40, -25, 0, 25, -40], [-26, 16.25, 0, -16.25, 26], [-8, 5, 0, -5, 8], [22, -13.75, 0, 13.75, -22], [-8, 5, 0, -5, 8]]) / 6.0
    lut = RectBivariateSpline(x, y, z)
    assert_array_almost_equal(lut.partial_derivative(1, 0)(x, y), dx)
    assert_array_almost_equal(lut.partial_derivative(0, 1)(x, y), dy)
    assert_array_almost_equal(lut.partial_derivative(1, 1)(x, y), dxdy)