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
def test_reentrant_func(self):

    def func(*args):
        self.test_basic()
        return self.residuals(*args)
    p0 = array([0, 0, 0])
    params_fit, ier = leastsq(func, p0, args=(self.y_meas, self.x))
    assert_(ier in (1, 2, 3, 4), 'solution not found (ier=%d)' % ier)
    assert_array_almost_equal(params_fit, self.abc, decimal=2)