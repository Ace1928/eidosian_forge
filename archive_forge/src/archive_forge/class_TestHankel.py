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
class TestHankel:

    def test_negv1(self):
        assert_almost_equal(special.hankel1(-3, 2), -special.hankel1(3, 2), 14)

    def test_hankel1(self):
        hank1 = special.hankel1(1, 0.1)
        hankrl = special.jv(1, 0.1) + special.yv(1, 0.1) * 1j
        assert_almost_equal(hank1, hankrl, 8)

    def test_negv1e(self):
        assert_almost_equal(special.hankel1e(-3, 2), -special.hankel1e(3, 2), 14)

    def test_hankel1e(self):
        hank1e = special.hankel1e(1, 0.1)
        hankrle = special.hankel1(1, 0.1) * exp(-0.1j)
        assert_almost_equal(hank1e, hankrle, 8)

    def test_negv2(self):
        assert_almost_equal(special.hankel2(-3, 2), -special.hankel2(3, 2), 14)

    def test_hankel2(self):
        hank2 = special.hankel2(1, 0.1)
        hankrl2 = special.jv(1, 0.1) - special.yv(1, 0.1) * 1j
        assert_almost_equal(hank2, hankrl2, 8)

    def test_neg2e(self):
        assert_almost_equal(special.hankel2e(-3, 2), -special.hankel2e(3, 2), 14)

    def test_hankl2e(self):
        hank2e = special.hankel2e(1, 0.1)
        hankrl2e = special.hankel2e(1, 0.1)
        assert_almost_equal(hank2e, hankrl2e, 8)