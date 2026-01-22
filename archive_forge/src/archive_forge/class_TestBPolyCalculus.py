from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
class TestBPolyCalculus:

    def test_derivative(self):
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        bp = BPoly(c, x)
        bp_der = bp.derivative()
        assert_allclose(bp_der(0.4), -6 * 0.6)
        assert_allclose(bp_der(1.7), 0.7)
        assert_allclose([bp(0.4, nu=1), bp(0.4, nu=2), bp(0.4, nu=3)], [-6 * (1 - 0.4), 6.0, 0.0])
        assert_allclose([bp(1.7, nu=1), bp(1.7, nu=2), bp(1.7, nu=3)], [0.7, 1.0, 0])

    def test_derivative_ppoly(self):
        np.random.seed(1234)
        m, k = (5, 8)
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m - 1))
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)
        for d in range(k):
            bp = bp.derivative()
            pp = pp.derivative()
            xp = np.linspace(x[0], x[-1], 21)
            assert_allclose(bp(xp), pp(xp))

    def test_deriv_inplace(self):
        np.random.seed(1234)
        m, k = (5, 8)
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m - 1))
        for cc in [c.copy(), c * (1.0 + 2j)]:
            bp = BPoly(cc, x)
            xp = np.linspace(x[0], x[-1], 21)
            for i in range(k):
                assert_allclose(bp(xp, i), bp.derivative(i)(xp))

    def test_antiderivative_simple(self):
        x = [0, 1, 3]
        c = [[0, 0], [1, 1]]
        bp = BPoly(c, x)
        bi = bp.antiderivative()
        xx = np.linspace(0, 3, 11)
        assert_allclose(bi(xx), np.where(xx < 1, xx ** 2 / 2.0, 0.5 * xx * (xx / 2.0 - 1) + 3.0 / 4), atol=1e-12, rtol=1e-12)

    def test_der_antider(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10, 2, 3))
        bp = BPoly(c, x)
        xx = np.linspace(x[0], x[-1], 100)
        assert_allclose(bp.antiderivative().derivative()(xx), bp(xx), atol=1e-12, rtol=1e-12)

    def test_antider_ppoly(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10, 2, 3))
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)
        xx = np.linspace(x[0], x[-1], 10)
        assert_allclose(bp.antiderivative(2)(xx), pp.antiderivative(2)(xx), atol=1e-12, rtol=1e-12)

    def test_antider_continuous(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10))
        bp = BPoly(c, x).antiderivative()
        xx = bp.x[1:-1]
        assert_allclose(bp(xx - 1e-14), bp(xx + 1e-14), atol=1e-12, rtol=1e-12)

    def test_integrate(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10))
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)
        assert_allclose(bp.integrate(0, 1), pp.integrate(0, 1), atol=1e-12, rtol=1e-12)

    def test_integrate_extrap(self):
        c = [[1]]
        x = [0, 1]
        b = BPoly(c, x)
        assert_allclose(b.integrate(0, 2), 2.0, atol=1e-14)
        b1 = BPoly(c, x, extrapolate=False)
        assert_(np.isnan(b1.integrate(0, 2)))
        assert_allclose(b1.integrate(0, 2, extrapolate=True), 2.0, atol=1e-14)

    def test_integrate_periodic(self):
        x = np.array([1, 2, 4])
        c = np.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
        P = BPoly.from_power_basis(PPoly(c, x), extrapolate='periodic')
        I = P.antiderivative()
        period_int = I(4) - I(1)
        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)
        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12), I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)
        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    def test_antider_neg(self):
        c = [[1]]
        x = [0, 1]
        b = BPoly(c, x)
        xx = np.linspace(0, 1, 21)
        assert_allclose(b.derivative(-1)(xx), b.antiderivative()(xx), atol=1e-12, rtol=1e-12)
        assert_allclose(b.derivative(1)(xx), b.antiderivative(-1)(xx), atol=1e-12, rtol=1e-12)