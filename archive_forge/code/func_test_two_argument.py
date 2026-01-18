import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def test_two_argument(self):

    def func(x, a, b):
        return b * x ** a
    popt, pcov = curve_fit(func, self.x, self.y)
    assert_(len(popt) == 2)
    assert_(pcov.shape == (2, 2))
    assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
    assert_array_almost_equal(pcov, [[0.0852, -0.126], [-0.126, 0.1912]], decimal=4)