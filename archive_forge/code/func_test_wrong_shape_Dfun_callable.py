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
def test_wrong_shape_Dfun_callable(self):
    func = ReturnShape(1)
    deriv_func = ReturnShape((2, 2))
    assert_raises(TypeError, optimize.leastsq, func, x0=[0, 1], Dfun=deriv_func)