import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def _numdiff_2d(func, x, y, dx=0, dy=0, eps=1e-08):
    if dx == 0 and dy == 0:
        return func(x, y)
    elif dx == 1 and dy == 0:
        return (func(x + eps, y) - func(x - eps, y)) / (2 * eps)
    elif dx == 0 and dy == 1:
        return (func(x, y + eps) - func(x, y - eps)) / (2 * eps)
    elif dx == 1 and dy == 1:
        return (func(x + eps, y + eps) - func(x - eps, y + eps) - func(x + eps, y - eps) + func(x - eps, y - eps)) / (2 * eps) ** 2
    else:
        raise ValueError('invalid derivative order')