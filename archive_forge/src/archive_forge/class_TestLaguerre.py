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
class TestLaguerre:

    def test_laguerre(self):
        lag0 = special.laguerre(0)
        lag1 = special.laguerre(1)
        lag2 = special.laguerre(2)
        lag3 = special.laguerre(3)
        lag4 = special.laguerre(4)
        lag5 = special.laguerre(5)
        assert_array_almost_equal(lag0.c, [1], 13)
        assert_array_almost_equal(lag1.c, [-1, 1], 13)
        assert_array_almost_equal(lag2.c, array([1, -4, 2]) / 2.0, 13)
        assert_array_almost_equal(lag3.c, array([-1, 9, -18, 6]) / 6.0, 13)
        assert_array_almost_equal(lag4.c, array([1, -16, 72, -96, 24]) / 24.0, 13)
        assert_array_almost_equal(lag5.c, array([-1, 25, -200, 600, -600, 120]) / 120.0, 13)

    def test_genlaguerre(self):
        k = 5 * np.random.random() - 0.9
        lag0 = special.genlaguerre(0, k)
        lag1 = special.genlaguerre(1, k)
        lag2 = special.genlaguerre(2, k)
        lag3 = special.genlaguerre(3, k)
        assert_equal(lag0.c, [1])
        assert_equal(lag1.c, [-1, k + 1])
        assert_almost_equal(lag2.c, array([1, -2 * (k + 2), (k + 1.0) * (k + 2.0)]) / 2.0)
        assert_almost_equal(lag3.c, array([-1, 3 * (k + 3), -3 * (k + 2) * (k + 3), (k + 1) * (k + 2) * (k + 3)]) / 6.0)