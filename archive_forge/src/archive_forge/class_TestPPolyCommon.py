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
class TestPPolyCommon:

    def test_sort_check(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 1, 0.5])
        assert_raises(ValueError, PPoly, c, x)
        assert_raises(ValueError, BPoly, c, x)

    def test_ctor_c(self):
        with assert_raises(ValueError):
            PPoly([1, 2], [0, 1])

    def test_extend(self):
        np.random.seed(1234)
        order = 3
        x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
        c = 2 * np.random.rand(order + 1, len(x) - 1, 2, 3) - 1
        for cls in (PPoly, BPoly):
            pp = cls(c[:, :9], x[:10])
            pp.extend(c[:, 9:], x[10:])
            pp2 = cls(c[:, 10:], x[10:])
            pp2.extend(c[:, :10], x[:10])
            pp3 = cls(c, x)
            assert_array_equal(pp.c, pp3.c)
            assert_array_equal(pp.x, pp3.x)
            assert_array_equal(pp2.c, pp3.c)
            assert_array_equal(pp2.x, pp3.x)

    def test_extend_diff_orders(self):
        np.random.seed(1234)
        x = np.linspace(0, 1, 6)
        c = np.random.rand(2, 5)
        x2 = np.linspace(1, 2, 6)
        c2 = np.random.rand(4, 5)
        for cls in (PPoly, BPoly):
            pp1 = cls(c, x)
            pp2 = cls(c2, x2)
            pp_comb = cls(c, x)
            pp_comb.extend(c2, x2[1:])
            xi1 = np.linspace(0, 1, 300, endpoint=False)
            xi2 = np.linspace(1, 2, 300)
            assert_allclose(pp1(xi1), pp_comb(xi1))
            assert_allclose(pp2(xi2), pp_comb(xi2))

    def test_extend_descending(self):
        np.random.seed(0)
        order = 3
        x = np.sort(np.random.uniform(0, 10, 20))
        c = np.random.rand(order + 1, x.shape[0] - 1, 2, 3)
        for cls in (PPoly, BPoly):
            p = cls(c, x)
            p1 = cls(c[:, :9], x[:10])
            p1.extend(c[:, 9:], x[10:])
            p2 = cls(c[:, 10:], x[10:])
            p2.extend(c[:, :10], x[:10])
            assert_array_equal(p1.c, p.c)
            assert_array_equal(p1.x, p.x)
            assert_array_equal(p2.c, p.c)
            assert_array_equal(p2.x, p.x)

    def test_shape(self):
        np.random.seed(1234)
        c = np.random.rand(8, 12, 5, 6, 7)
        x = np.sort(np.random.rand(13))
        xp = np.random.rand(3, 4)
        for cls in (PPoly, BPoly):
            p = cls(c, x)
            assert_equal(p(xp).shape, (3, 4, 5, 6, 7))
        for cls in (PPoly, BPoly):
            p = cls(c[..., 0, 0, 0], x)
            assert_equal(np.shape(p(0.5)), ())
            assert_equal(np.shape(p(np.array(0.5))), ())
            assert_raises(ValueError, p, np.array([[0.1, 0.2], [0.4]], dtype=object))

    def test_complex_coef(self):
        np.random.seed(12345)
        x = np.sort(np.random.random(13))
        c = np.random.random((8, 12)) * (1.0 + 0.3j)
        c_re, c_im = (c.real, c.imag)
        xp = np.random.random(5)
        for cls in (PPoly, BPoly):
            p, p_re, p_im = (cls(c, x), cls(c_re, x), cls(c_im, x))
            for nu in [0, 1, 2]:
                assert_allclose(p(xp, nu).real, p_re(xp, nu))
                assert_allclose(p(xp, nu).imag, p_im(xp, nu))

    def test_axis(self):
        np.random.seed(12345)
        c = np.random.rand(3, 4, 5, 6, 7, 8)
        c_s = c.shape
        xp = np.random.random((1, 2))
        for axis in (0, 1, 2, 3):
            m = c.shape[axis + 1]
            x = np.sort(np.random.rand(m + 1))
            for cls in (PPoly, BPoly):
                p = cls(c, x, axis=axis)
                assert_equal(p.c.shape, c_s[axis:axis + 2] + c_s[:axis] + c_s[axis + 2:])
                res = p(xp)
                targ_shape = c_s[:axis] + xp.shape + c_s[2 + axis:]
                assert_equal(res.shape, targ_shape)
                for p1 in [cls(c, x, axis=axis).derivative(), cls(c, x, axis=axis).derivative(2), cls(c, x, axis=axis).antiderivative(), cls(c, x, axis=axis).antiderivative(2)]:
                    assert_equal(p1.axis, p.axis)
        for axis in (-1, 4, 5, 6):
            for cls in (BPoly, PPoly):
                assert_raises(ValueError, cls, **dict(c=c, x=x, axis=axis))