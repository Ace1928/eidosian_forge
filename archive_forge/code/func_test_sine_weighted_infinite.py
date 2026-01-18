import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,
import pytest
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes
import scipy.integrate._test_multivariate as clib_test
def test_sine_weighted_infinite(self):

    def myfunc(x, a):
        return exp(-x * a)
    a = 4.0
    ome = 3.0
    assert_quad(quad(myfunc, 0, np.inf, args=a, weight='sin', wvar=ome), ome / (a ** 2 + ome ** 2))