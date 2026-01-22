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
class TestStruve:

    def _series(self, v, z, n=100):
        """Compute Struve function & error estimate from its power series."""
        k = arange(0, n)
        r = (-1) ** k * (0.5 * z) ** (2 * k + v + 1) / special.gamma(k + 1.5) / special.gamma(k + v + 1.5)
        err = abs(r).max() * finfo(double).eps * n
        return (r.sum(), err)

    def test_vs_series(self):
        """Check Struve function versus its power series"""
        for v in [-20, -10, -7.99, -3.4, -1, 0, 1, 3.4, 12.49, 16]:
            for z in [1, 10, 19, 21, 30]:
                value, err = self._series(v, z)
                (assert_allclose(special.struve(v, z), value, rtol=0, atol=err), (v, z))

    def test_some_values(self):
        assert_allclose(special.struve(-7.99, 21), 0.0467547614113, rtol=1e-07)
        assert_allclose(special.struve(-8.01, 21), 0.0398716951023, rtol=1e-08)
        assert_allclose(special.struve(-3.0, 200), 0.0142134427432, rtol=1e-12)
        assert_allclose(special.struve(-8.0, -41), 0.0192469727846, rtol=1e-11)
        assert_equal(special.struve(-12, -41), -special.struve(-12, 41))
        assert_equal(special.struve(+12, -41), -special.struve(+12, 41))
        assert_equal(special.struve(-11, -41), +special.struve(-11, 41))
        assert_equal(special.struve(+11, -41), +special.struve(+11, 41))
        assert_(isnan(special.struve(-7.1, -1)))
        assert_(isnan(special.struve(-10.1, -1)))

    def test_regression_679(self):
        """Regression test for #679"""
        assert_allclose(special.struve(-1.0, 20 - 1e-08), special.struve(-1.0, 20 + 1e-08))
        assert_allclose(special.struve(-2.0, 20 - 1e-08), special.struve(-2.0, 20 + 1e-08))
        assert_allclose(special.struve(-4.3, 20 - 1e-08), special.struve(-4.3, 20 + 1e-08))