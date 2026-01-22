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
class TestBetaInc:
    """
    Tests for betainc, betaincinv, betaincc, betainccinv.
    """

    def test_a1_b1(self):
        x = np.array([0, 0.25, 1])
        assert_equal(special.betainc(1, 1, x), x)
        assert_equal(special.betaincinv(1, 1, x), x)
        assert_equal(special.betaincc(1, 1, x), 1 - x)
        assert_equal(special.betainccinv(1, 1, x), 1 - x)

    @pytest.mark.parametrize('a, b, x, p', [(2, 4, 0.3138101704556974, 0.5), (0.0342, 171.0, 1e-10, 0.5526991690180709), (0.0342, 171, 8.42313169354797e-21, 0.25), (0.0002742794749792665, 289206.03125, 1.639984034231756e-56, 0.9688708782196045), (4, 99997, 0.0001947841578892121, 0.999995)])
    def test_betainc_betaincinv(self, a, b, x, p):
        p1 = special.betainc(a, b, x)
        assert_allclose(p1, p, rtol=1e-15)
        x1 = special.betaincinv(a, b, p)
        assert_allclose(x1, x, rtol=5e-13)

    @pytest.mark.parametrize('a, b, x, p', [(2.5, 3.0, 0.25, 0.833251953125), (7.5, 13.25, 0.375, 0.43298734645560366), (0.125, 7.5, 0.425, 0.0006688257851314237), (0.125, 18.0, 1e-06, 0.7298235914509633), (0.125, 18.0, 0.996, 7.274587553838015e-46), (0.125, 24.0, 0.75, 3.70853404816862e-17), (16.0, 0.75, 0.99999999975, 5.440875927741863e-07), (0.4211959643503401, 16939.046996018118, 0.000815296167195521, 1e-07)])
    def test_betaincc_betainccinv(self, a, b, x, p):
        p1 = special.betaincc(a, b, x)
        assert_allclose(p1, p, rtol=5e-15)
        x1 = special.betainccinv(a, b, p)
        assert_allclose(x1, x, rtol=8e-15)

    @pytest.mark.parametrize('a, b, y, ref', [(14.208308325339239, 14.208308325339239, 7.703145458496392e-307, 8.566004561846704e-23), (14.0, 14.5, 1e-280, 2.9343915006642424e-21), (3.5, 15.0, 4e-95, 1.3290751429289227e-28), (10.0, 1.25, 2e-234, 3.982659092143654e-24), (4.0, 99997.0, 5e-88, 3.309800566862242e-27)])
    def test_betaincinv_tiny_y(self, a, b, y, ref):
        x = special.betaincinv(a, b, y)
        assert_allclose(x, ref, rtol=1e-14)

    @pytest.mark.parametrize('func', [special.betainc, special.betaincinv, special.betaincc, special.betainccinv])
    @pytest.mark.parametrize('args', [(-1.0, 2, 0.5), (0, 2, 0.5), (1.5, -2.0, 0.5), (1.5, 0, 0.5), (1.5, 2.0, -0.3), (1.5, 2.0, 1.1)])
    def test_betainc_domain_errors(self, func, args):
        with special.errstate(domain='raise'):
            with pytest.raises(special.SpecialFunctionError, match='domain'):
                special.betainc(*args)