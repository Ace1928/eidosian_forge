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
class TestNdPPoly:

    def test_simple_1d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5)
        x = np.linspace(0, 1, 5 + 1)
        xi = np.random.rand(200)
        p = NdPPoly(c, (x,))
        v1 = p((xi,))
        v2 = _ppoly_eval_1(c[:, :, None], x, xi).ravel()
        assert_allclose(v1, v2)

    def test_simple_2d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 6, 7)
        x = np.linspace(0, 1, 6 + 1)
        y = np.linspace(0, 1, 7 + 1) ** 2
        xi = np.random.rand(200)
        yi = np.random.rand(200)
        v1 = np.empty([len(xi), 1], dtype=c.dtype)
        v1.fill(np.nan)
        _ppoly.evaluate_nd(c.reshape(4 * 5, 6 * 7, 1), (x, y), np.array([4, 5], dtype=np.intc), np.c_[xi, yi], np.array([0, 0], dtype=np.intc), 1, v1)
        v1 = v1.ravel()
        v2 = _ppoly2d_eval(c, (x, y), xi, yi)
        assert_allclose(v1, v2)
        p = NdPPoly(c, (x, y))
        for nu in (None, (0, 0), (0, 1), (1, 0), (2, 3), (9, 2)):
            v1 = p(np.c_[xi, yi], nu=nu)
            v2 = _ppoly2d_eval(c, (x, y), xi, yi, nu=nu)
            assert_allclose(v1, v2, err_msg=repr(nu))

    def test_simple_3d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 6, 7, 8, 9)
        x = np.linspace(0, 1, 7 + 1)
        y = np.linspace(0, 1, 8 + 1) ** 2
        z = np.linspace(0, 1, 9 + 1) ** 3
        xi = np.random.rand(40)
        yi = np.random.rand(40)
        zi = np.random.rand(40)
        p = NdPPoly(c, (x, y, z))
        for nu in (None, (0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 3, 0), (6, 0, 2)):
            v1 = p((xi, yi, zi), nu=nu)
            v2 = _ppoly3d_eval(c, (x, y, z), xi, yi, zi, nu=nu)
            assert_allclose(v1, v2, err_msg=repr(nu))

    def test_simple_4d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 6, 7, 8, 9, 10, 11)
        x = np.linspace(0, 1, 8 + 1)
        y = np.linspace(0, 1, 9 + 1) ** 2
        z = np.linspace(0, 1, 10 + 1) ** 3
        u = np.linspace(0, 1, 11 + 1) ** 4
        xi = np.random.rand(20)
        yi = np.random.rand(20)
        zi = np.random.rand(20)
        ui = np.random.rand(20)
        p = NdPPoly(c, (x, y, z, u))
        v1 = p((xi, yi, zi, ui))
        v2 = _ppoly4d_eval(c, (x, y, z, u), xi, yi, zi, ui)
        assert_allclose(v1, v2)

    def test_deriv_1d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5)
        x = np.linspace(0, 1, 5 + 1)
        p = NdPPoly(c, (x,))
        dp = p.derivative(nu=[1])
        p1 = PPoly(c, x)
        dp1 = p1.derivative()
        assert_allclose(dp.c, dp1.c)
        dp = p.antiderivative(nu=[2])
        p1 = PPoly(c, x)
        dp1 = p1.antiderivative(2)
        assert_allclose(dp.c, dp1.c)

    def test_deriv_3d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 6, 7, 8, 9)
        x = np.linspace(0, 1, 7 + 1)
        y = np.linspace(0, 1, 8 + 1) ** 2
        z = np.linspace(0, 1, 9 + 1) ** 3
        p = NdPPoly(c, (x, y, z))
        p1 = PPoly(c.transpose(0, 3, 1, 2, 4, 5), x)
        dp = p.derivative(nu=[2])
        dp1 = p1.derivative(2)
        assert_allclose(dp.c, dp1.c.transpose(0, 2, 3, 1, 4, 5))
        p1 = PPoly(c.transpose(1, 4, 0, 2, 3, 5), y)
        dp = p.antiderivative(nu=[0, 1, 0])
        dp1 = p1.antiderivative(1)
        assert_allclose(dp.c, dp1.c.transpose(2, 0, 3, 4, 1, 5))
        p1 = PPoly(c.transpose(2, 5, 0, 1, 3, 4), z)
        dp = p.derivative(nu=[0, 0, 3])
        dp1 = p1.derivative(3)
        assert_allclose(dp.c, dp1.c.transpose(2, 3, 0, 4, 5, 1))

    def test_deriv_3d_simple(self):
        c = np.ones((1, 1, 1, 3, 4, 5))
        x = np.linspace(0, 1, 3 + 1) ** 1
        y = np.linspace(0, 1, 4 + 1) ** 2
        z = np.linspace(0, 1, 5 + 1) ** 3
        p = NdPPoly(c, (x, y, z))
        ip = p.antiderivative((1, 0, 4))
        ip = ip.antiderivative((0, 2, 0))
        xi = np.random.rand(20)
        yi = np.random.rand(20)
        zi = np.random.rand(20)
        assert_allclose(ip((xi, yi, zi)), xi * yi ** 2 * zi ** 4 / (gamma(3) * gamma(5)))

    def test_integrate_2d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 16, 17)
        x = np.linspace(0, 1, 16 + 1) ** 1
        y = np.linspace(0, 1, 17 + 1) ** 2
        c = c.transpose(0, 2, 1, 3)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        _ppoly.fix_continuity(cx, x, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(0, 2, 1, 3)
        c = c.transpose(1, 3, 0, 2)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        _ppoly.fix_continuity(cx, y, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(2, 0, 3, 1).copy()
        p = NdPPoly(c, (x, y))
        for ranges in [[(0, 1), (0, 1)], [(0, 0.5), (0, 1)], [(0, 1), (0, 0.5)], [(0.3, 0.7), (0.6, 0.2)]]:
            ig = p.integrate(ranges)
            ig2, err2 = nquad(lambda x, y: p((x, y)), ranges, opts=[dict(epsrel=1e-05, epsabs=1e-05)] * 2)
            assert_allclose(ig, ig2, rtol=1e-05, atol=1e-05, err_msg=repr(ranges))

    def test_integrate_1d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 6, 16, 17, 18)
        x = np.linspace(0, 1, 16 + 1) ** 1
        y = np.linspace(0, 1, 17 + 1) ** 2
        z = np.linspace(0, 1, 18 + 1) ** 3
        p = NdPPoly(c, (x, y, z))
        u = np.random.rand(200)
        v = np.random.rand(200)
        a, b = (0.2, 0.7)
        px = p.integrate_1d(a, b, axis=0)
        pax = p.antiderivative((1, 0, 0))
        assert_allclose(px((u, v)), pax((b, u, v)) - pax((a, u, v)))
        py = p.integrate_1d(a, b, axis=1)
        pay = p.antiderivative((0, 1, 0))
        assert_allclose(py((u, v)), pay((u, b, v)) - pay((u, a, v)))
        pz = p.integrate_1d(a, b, axis=2)
        paz = p.antiderivative((0, 0, 1))
        assert_allclose(pz((u, v)), paz((u, v, b)) - paz((u, v, a)))