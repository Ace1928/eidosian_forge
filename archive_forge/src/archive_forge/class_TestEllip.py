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
class TestEllip:

    def test_ellipj_nan(self):
        """Regression test for #912."""
        special.ellipj(0.5, np.nan)

    def test_ellipj(self):
        el = special.ellipj(0.2, 0)
        rel = [sin(0.2), cos(0.2), 1.0, 0.2]
        assert_array_almost_equal(el, rel, 13)

    def test_ellipk(self):
        elk = special.ellipk(0.2)
        assert_almost_equal(elk, 1.659623598610528, 11)
        assert_equal(special.ellipkm1(0.0), np.inf)
        assert_equal(special.ellipkm1(1.0), pi / 2)
        assert_equal(special.ellipkm1(np.inf), 0.0)
        assert_equal(special.ellipkm1(np.nan), np.nan)
        assert_equal(special.ellipkm1(-1), np.nan)
        assert_allclose(special.ellipk(-10), 0.7908718902387385)

    def test_ellipkinc(self):
        elkinc = special.ellipkinc(pi / 2, 0.2)
        elk = special.ellipk(0.2)
        assert_almost_equal(elkinc, elk, 15)
        alpha = 20 * pi / 180
        phi = 45 * pi / 180
        m = sin(alpha) ** 2
        elkinc = special.ellipkinc(phi, m)
        assert_almost_equal(elkinc, 0.79398143, 8)
        assert_equal(special.ellipkinc(pi / 2, 0.0), pi / 2)
        assert_equal(special.ellipkinc(pi / 2, 1.0), np.inf)
        assert_equal(special.ellipkinc(pi / 2, -np.inf), 0.0)
        assert_equal(special.ellipkinc(pi / 2, np.nan), np.nan)
        assert_equal(special.ellipkinc(pi / 2, 2), np.nan)
        assert_equal(special.ellipkinc(0, 0.5), 0.0)
        assert_equal(special.ellipkinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipkinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipkinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipkinc(np.nan, np.nan), np.nan)
        assert_allclose(special.ellipkinc(0.3897411203531872, 1), 0.4, rtol=1e-14)
        assert_allclose(special.ellipkinc(1.5707, -10), 0.7908428466172495)

    def test_ellipkinc_2(self):
        mbad = 0.6835937500000001
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)
        mvals = []
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        f = special.ellipkinc(phi, mvals)
        assert_array_almost_equal_nulp(f, np.full_like(f, 1.0259330100195334), 1)
        f1 = special.ellipkinc(phi + pi, mvals)
        assert_array_almost_equal_nulp(f1, np.full_like(f1, 5.1296650500976675), 2)

    def test_ellipkinc_singular(self):
        xlog = np.logspace(-300, -17, 25)
        xlin = np.linspace(1e-17, 0.1, 25)
        xlin2 = np.linspace(0.1, pi / 2, 25, endpoint=False)
        assert_allclose(special.ellipkinc(xlog, 1), np.arcsinh(np.tan(xlog)), rtol=100000000000000.0)
        assert_allclose(special.ellipkinc(xlin, 1), np.arcsinh(np.tan(xlin)), rtol=100000000000000.0)
        assert_allclose(special.ellipkinc(xlin2, 1), np.arcsinh(np.tan(xlin2)), rtol=100000000000000.0)
        assert_equal(special.ellipkinc(np.pi / 2, 1), np.inf)
        assert_allclose(special.ellipkinc(-xlog, 1), np.arcsinh(np.tan(-xlog)), rtol=100000000000000.0)
        assert_allclose(special.ellipkinc(-xlin, 1), np.arcsinh(np.tan(-xlin)), rtol=100000000000000.0)
        assert_allclose(special.ellipkinc(-xlin2, 1), np.arcsinh(np.tan(-xlin2)), rtol=100000000000000.0)
        assert_equal(special.ellipkinc(-np.pi / 2, 1), np.inf)

    def test_ellipe(self):
        ele = special.ellipe(0.2)
        assert_almost_equal(ele, 1.489035058095853, 8)
        assert_equal(special.ellipe(0.0), pi / 2)
        assert_equal(special.ellipe(1.0), 1.0)
        assert_equal(special.ellipe(-np.inf), np.inf)
        assert_equal(special.ellipe(np.nan), np.nan)
        assert_equal(special.ellipe(2), np.nan)
        assert_allclose(special.ellipe(-10), 3.639138038417769)

    def test_ellipeinc(self):
        eleinc = special.ellipeinc(pi / 2, 0.2)
        ele = special.ellipe(0.2)
        assert_almost_equal(eleinc, ele, 14)
        alpha, phi = (52 * pi / 180, 35 * pi / 180)
        m = sin(alpha) ** 2
        eleinc = special.ellipeinc(phi, m)
        assert_almost_equal(eleinc, 0.58823065, 8)
        assert_equal(special.ellipeinc(pi / 2, 0.0), pi / 2)
        assert_equal(special.ellipeinc(pi / 2, 1.0), 1.0)
        assert_equal(special.ellipeinc(pi / 2, -np.inf), np.inf)
        assert_equal(special.ellipeinc(pi / 2, np.nan), np.nan)
        assert_equal(special.ellipeinc(pi / 2, 2), np.nan)
        assert_equal(special.ellipeinc(0, 0.5), 0.0)
        assert_equal(special.ellipeinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipeinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipeinc(np.inf, -np.inf), np.inf)
        assert_equal(special.ellipeinc(-np.inf, -np.inf), -np.inf)
        assert_equal(special.ellipeinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipeinc(np.nan, np.nan), np.nan)
        assert_allclose(special.ellipeinc(1.5707, -10), 3.6388185585822876)

    def test_ellipeinc_2(self):
        mbad = 0.6835937500000001
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)
        mvals = []
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        f = special.ellipeinc(phi, mvals)
        assert_array_almost_equal_nulp(f, np.full_like(f, 0.8444288457478102), 2)
        f1 = special.ellipeinc(phi + pi, mvals)
        assert_array_almost_equal_nulp(f1, np.full_like(f1, 3.347144228739051), 4)