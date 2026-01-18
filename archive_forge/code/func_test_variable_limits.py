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
def test_variable_limits(self):
    scale = 0.1

    def func2(x0, x1, x2, x3, t0, t1):
        val = x0 * x1 * x3 ** 2 + np.sin(x2) + 1 + (1 if x0 + t1 * x1 - t0 > 0 else 0)
        return val

    def lim0(x1, x2, x3, t0, t1):
        return [scale * (x1 ** 2 + x2 + np.cos(x3) * t0 * t1 + 1) - 1, scale * (x1 ** 2 + x2 + np.cos(x3) * t0 * t1 + 1) + 1]

    def lim1(x2, x3, t0, t1):
        return [scale * (t0 * x2 + t1 * x3) - 1, scale * (t0 * x2 + t1 * x3) + 1]

    def lim2(x3, t0, t1):
        return [scale * (x3 + t0 ** 2 * t1 ** 3) - 1, scale * (x3 + t0 ** 2 * t1 ** 3) + 1]

    def lim3(t0, t1):
        return [scale * (t0 + t1) - 1, scale * (t0 + t1) + 1]

    def opts0(x1, x2, x3, t0, t1):
        return {'points': [t0 - t1 * x1]}

    def opts1(x2, x3, t0, t1):
        return {}

    def opts2(x3, t0, t1):
        return {}

    def opts3(t0, t1):
        return {}
    res = nquad(func2, [lim0, lim1, lim2, lim3], args=(0, 0), opts=[opts0, opts1, opts2, opts3])
    assert_quad(res, 25.066666666666663)