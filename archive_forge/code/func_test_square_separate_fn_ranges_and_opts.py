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
def test_square_separate_fn_ranges_and_opts(self):

    def f(y, x):
        return 1.0

    def fn_range0(*args):
        return (-1, 1)

    def fn_range1(*args):
        return (-1, 1)

    def fn_opt0(*args):
        return {}

    def fn_opt1(*args):
        return {}
    ranges = [fn_range0, fn_range1]
    opts = [fn_opt0, fn_opt1]
    assert_quad(nquad(f, ranges, opts=opts), 4.0)