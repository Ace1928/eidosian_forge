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
class ReturnShape:
    """This class exists to create a callable that does not have a '__name__' attribute.

    __init__ takes the argument 'shape', which should be a tuple of ints.
    When an instance is called with a single argument 'x', it returns numpy.ones(shape).
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return np.ones(self.shape)