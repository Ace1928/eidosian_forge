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
class TestBPoly:

    def test_simple(self):
        x = [0, 1]
        c = [[3]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.1), 3.0)

    def test_simple2(self):
        x = [0, 1]
        c = [[3], [1]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.1), 3 * 0.9 + 1.0 * 0.1)

    def test_simple3(self):
        x = [0, 1]
        c = [[3], [1], [4]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.2), 3 * 0.8 * 0.8 + 1 * 2 * 0.2 * 0.8 + 4 * 0.2 * 0.2)

    def test_simple4(self):
        x = [0, 1]
        c = [[1], [1], [1], [2]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.3), 0.7 ** 3 + 3 * 0.7 ** 2 * 0.3 + 3 * 0.7 * 0.3 ** 2 + 2 * 0.3 ** 3)

    def test_simple5(self):
        x = [0, 1]
        c = [[1], [1], [8], [2], [1]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.3), 0.7 ** 4 + 4 * 0.7 ** 3 * 0.3 + 8 * 6 * 0.7 ** 2 * 0.3 ** 2 + 2 * 4 * 0.7 * 0.3 ** 3 + 0.3 ** 4)

    def test_periodic(self):
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        bp = BPoly(c, x, extrapolate='periodic')
        assert_allclose(bp(3.4), 3 * 0.6 ** 2)
        assert_allclose(bp(-1.3), 2 * (0.7 / 2) ** 2)
        assert_allclose(bp(3.4, 1), -6 * 0.6)
        assert_allclose(bp(-1.3, 1), 2 * (0.7 / 2))

    def test_descending(self):
        np.random.seed(0)
        power = 3
        for m in [10, 20, 30]:
            x = np.sort(np.random.uniform(0, 10, m + 1))
            ca = np.random.uniform(-0.1, 0.1, size=(power + 1, m))
            cd = ca[::-1].copy()
            pa = BPoly(ca, x, extrapolate=True)
            pd = BPoly(cd[:, ::-1], x[::-1], extrapolate=True)
            x_test = np.random.uniform(-10, 20, 100)
            assert_allclose(pa(x_test), pd(x_test), rtol=1e-13)
            assert_allclose(pa(x_test, 1), pd(x_test, 1), rtol=1e-13)
            pa_d = pa.derivative()
            pd_d = pd.derivative()
            assert_allclose(pa_d(x_test), pd_d(x_test), rtol=1e-13)
            pa_i = pa.antiderivative()
            pd_i = pd.antiderivative()
            for a, b in np.random.uniform(-10, 20, (5, 2)):
                int_a = pa.integrate(a, b)
                int_d = pd.integrate(a, b)
                assert_allclose(int_a, int_d, rtol=1e-12)
                assert_allclose(pa_i(b) - pa_i(a), pd_i(b) - pd_i(a), rtol=1e-12)

    def test_multi_shape(self):
        c = np.random.rand(6, 2, 1, 2, 3)
        x = np.array([0, 0.5, 1])
        p = BPoly(c, x)
        assert_equal(p.x.shape, x.shape)
        assert_equal(p.c.shape, c.shape)
        assert_equal(p(0.3).shape, c.shape[2:])
        assert_equal(p(np.random.rand(5, 6)).shape, (5, 6) + c.shape[2:])
        dp = p.derivative()
        assert_equal(dp.c.shape, (5, 2, 1, 2, 3))

    def test_interval_length(self):
        x = [0, 2]
        c = [[3], [1], [4]]
        bp = BPoly(c, x)
        xval = 0.1
        s = xval / 2
        assert_allclose(bp(xval), 3 * (1 - s) * (1 - s) + 1 * 2 * s * (1 - s) + 4 * s * s)

    def test_two_intervals(self):
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.4), 3 * 0.6 * 0.6)
        assert_allclose(bp(1.7), 2 * (0.7 / 2) ** 2)

    def test_extrapolate_attr(self):
        x = [0, 2]
        c = [[3], [1], [4]]
        bp = BPoly(c, x)
        for extrapolate in (True, False, None):
            bp = BPoly(c, x, extrapolate=extrapolate)
            bp_d = bp.derivative()
            if extrapolate is False:
                assert_(np.isnan(bp([-0.1, 2.1])).all())
                assert_(np.isnan(bp_d([-0.1, 2.1])).all())
            else:
                assert_(not np.isnan(bp([-0.1, 2.1])).any())
                assert_(not np.isnan(bp_d([-0.1, 2.1])).any())