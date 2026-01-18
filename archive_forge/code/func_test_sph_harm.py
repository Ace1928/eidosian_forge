import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
def test_sph_harm():
    sh = special.sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    assert_array_almost_equal(sh(0, 0, 0, 0), 0.5 / sqrt(pi))
    assert_array_almost_equal(sh(-2, 2, 0.0, pi / 4), 0.25 * sqrt(15.0 / (2.0 * pi)) * sin(pi / 4) ** 2.0)
    assert_array_almost_equal(sh(-2, 2, 0.0, pi / 2), 0.25 * sqrt(15.0 / (2.0 * pi)))
    assert_array_almost_equal(sh(2, 2, pi, pi / 2), 0.25 * sqrt(15 / (2.0 * pi)) * exp(0 + 2.0 * pi * 1j) * sin(pi / 2.0) ** 2.0)
    assert_array_almost_equal(sh(2, 4, pi / 4.0, pi / 3.0), 3.0 / 8.0 * sqrt(5.0 / (2.0 * pi)) * exp(0 + 2.0 * pi / 4.0 * 1j) * sin(pi / 3.0) ** 2.0 * (7.0 * cos(pi / 3.0) ** 2.0 - 1))
    assert_array_almost_equal(sh(4, 4, pi / 8.0, pi / 6.0), 3.0 / 16.0 * sqrt(35.0 / (2.0 * pi)) * exp(0 + 4.0 * pi / 8.0 * 1j) * sin(pi / 6.0) ** 4.0)