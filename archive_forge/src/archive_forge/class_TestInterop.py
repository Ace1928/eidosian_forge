import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
class TestInterop:

    def setup_method(self):
        xx = np.linspace(0, 4.0 * np.pi, 41)
        yy = np.cos(xx)
        b = make_interp_spline(xx, yy)
        self.tck = (b.t, b.c, b.k)
        self.xx, self.yy, self.b = (xx, yy, b)
        self.xnew = np.linspace(0, 4.0 * np.pi, 21)
        c2 = np.c_[b.c, b.c, b.c]
        self.c2 = np.dstack((c2, c2))
        self.b2 = BSpline(b.t, self.c2, b.k)

    def test_splev(self):
        xnew, b, b2 = (self.xnew, self.b, self.b2)
        assert_allclose(splev(xnew, b), b(xnew), atol=1e-15, rtol=1e-15)
        assert_allclose(splev(xnew, b.tck), b(xnew), atol=1e-15, rtol=1e-15)
        assert_allclose([splev(x, b) for x in xnew], b(xnew), atol=1e-15, rtol=1e-15)
        with assert_raises(ValueError, match='Calling splev.. with BSpline'):
            splev(xnew, b2)
        sh = tuple(range(1, b2.c.ndim)) + (0,)
        cc = b2.c.transpose(sh)
        tck = (b2.t, cc, b2.k)
        assert_allclose(splev(xnew, tck), b2(xnew).transpose(sh), atol=1e-15, rtol=1e-15)

    def test_splrep(self):
        x, y = (self.xx, self.yy)
        tck = splrep(x, y)
        t, c, k = _impl.splrep(x, y)
        assert_allclose(tck[0], t, atol=1e-15)
        assert_allclose(tck[1], c, atol=1e-15)
        assert_equal(tck[2], k)
        tck_f, _, _, _ = splrep(x, y, full_output=True)
        assert_allclose(tck_f[0], t, atol=1e-15)
        assert_allclose(tck_f[1], c, atol=1e-15)
        assert_equal(tck_f[2], k)
        yy = splev(x, tck)
        assert_allclose(y, yy, atol=1e-15)
        b = BSpline(*tck)
        assert_allclose(y, b(x), atol=1e-15)

    def test_splrep_errors(self):
        x, y = (self.xx, self.yy)
        y2 = np.c_[y, y]
        with assert_raises(ValueError):
            splrep(x, y2)
        with assert_raises(ValueError):
            _impl.splrep(x, y2)
        with assert_raises(TypeError, match='m > k must hold'):
            splrep(x[:3], y[:3])
        with assert_raises(TypeError, match='m > k must hold'):
            _impl.splrep(x[:3], y[:3])

    def test_splprep(self):
        x = np.arange(15).reshape((3, 5))
        b, u = splprep(x)
        tck, u1 = _impl.splprep(x)
        assert_allclose(u, u1, atol=1e-15)
        assert_allclose(splev(u, b), x, atol=1e-15)
        assert_allclose(splev(u, tck), x, atol=1e-15)
        (b_f, u_f), _, _, _ = splprep(x, s=0, full_output=True)
        assert_allclose(u, u_f, atol=1e-15)
        assert_allclose(splev(u_f, b_f), x, atol=1e-15)

    def test_splprep_errors(self):
        x = np.arange(3 * 4 * 5).reshape((3, 4, 5))
        with assert_raises(ValueError, match='too many values to unpack'):
            splprep(x)
        with assert_raises(ValueError, match='too many values to unpack'):
            _impl.splprep(x)
        x = np.linspace(0, 40, num=3)
        with assert_raises(TypeError, match='m > k must hold'):
            splprep([x])
        with assert_raises(TypeError, match='m > k must hold'):
            _impl.splprep([x])
        x = [-50.49072266, -50.49072266, -54.49072266, -54.49072266]
        with assert_raises(ValueError, match='Invalid inputs'):
            splprep([x])
        with assert_raises(ValueError, match='Invalid inputs'):
            _impl.splprep([x])
        x = [1, 3, 2, 4]
        u = [0, 0.3, 0.2, 1]
        with assert_raises(ValueError, match='Invalid inputs'):
            splprep(*[[x], None, u])

    def test_sproot(self):
        b, b2 = (self.b, self.b2)
        roots = np.array([0.5, 1.5, 2.5, 3.5]) * np.pi
        assert_allclose(sproot(b), roots, atol=1e-07, rtol=1e-07)
        assert_allclose(sproot((b.t, b.c, b.k)), roots, atol=1e-07, rtol=1e-07)
        with assert_raises(ValueError, match='Calling sproot.. with BSpline'):
            sproot(b2, mest=50)
        c2r = b2.c.transpose(1, 2, 0)
        rr = np.asarray(sproot((b2.t, c2r, b2.k), mest=50))
        assert_equal(rr.shape, (3, 2, 4))
        assert_allclose(rr - roots, 0, atol=1e-12)

    def test_splint(self):
        b, b2 = (self.b, self.b2)
        assert_allclose(splint(0, 1, b), splint(0, 1, b.tck), atol=1e-14)
        assert_allclose(splint(0, 1, b), b.integrate(0, 1), atol=1e-14)
        with assert_raises(ValueError, match='Calling splint.. with BSpline'):
            splint(0, 1, b2)
        c2r = b2.c.transpose(1, 2, 0)
        integr = np.asarray(splint(0, 1, (b2.t, c2r, b2.k)))
        assert_equal(integr.shape, (3, 2))
        assert_allclose(integr, splint(0, 1, b), atol=1e-14)

    def test_splder(self):
        for b in [self.b, self.b2]:
            ct = len(b.t) - len(b.c)
            if ct > 0:
                b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]
            for n in [1, 2, 3]:
                bd = splder(b)
                tck_d = _impl.splder((b.t, b.c, b.k))
                assert_allclose(bd.t, tck_d[0], atol=1e-15)
                assert_allclose(bd.c, tck_d[1], atol=1e-15)
                assert_equal(bd.k, tck_d[2])
                assert_(isinstance(bd, BSpline))
                assert_(isinstance(tck_d, tuple))

    def test_splantider(self):
        for b in [self.b, self.b2]:
            ct = len(b.t) - len(b.c)
            if ct > 0:
                b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]
            for n in [1, 2, 3]:
                bd = splantider(b)
                tck_d = _impl.splantider((b.t, b.c, b.k))
                assert_allclose(bd.t, tck_d[0], atol=1e-15)
                assert_allclose(bd.c, tck_d[1], atol=1e-15)
                assert_equal(bd.k, tck_d[2])
                assert_(isinstance(bd, BSpline))
                assert_(isinstance(tck_d, tuple))

    def test_insert(self):
        b, b2, xx = (self.b, self.b2, self.xx)
        j = b.t.size // 2
        tn = 0.5 * (b.t[j] + b.t[j + 1])
        bn, tck_n = (insert(tn, b), insert(tn, (b.t, b.c, b.k)))
        assert_allclose(splev(xx, bn), splev(xx, tck_n), atol=1e-15)
        assert_(isinstance(bn, BSpline))
        assert_(isinstance(tck_n, tuple))
        sh = tuple(range(b2.c.ndim))
        c_ = b2.c.transpose(sh[1:] + (0,))
        tck_n2 = insert(tn, (b2.t, c_, b2.k))
        bn2 = insert(tn, b2)
        assert_allclose(np.asarray(splev(xx, tck_n2)).transpose(2, 0, 1), bn2(xx), atol=1e-15)
        assert_(isinstance(bn2, BSpline))
        assert_(isinstance(tck_n2, tuple))