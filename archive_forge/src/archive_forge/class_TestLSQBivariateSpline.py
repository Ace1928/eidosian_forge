import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestLSQBivariateSpline:

    def test_linear_constant(self):
        x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        s = 0.1
        tx = [1 + s, 3 - s]
        ty = [1 + s, 3 - s]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, '\nThe coefficients of the spline')
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
            assert_equal(len(r), 1)
        assert_almost_equal(lut(2, 2), 3.0)

    def test_bilinearity(self):
        x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = [0, 7, 8, 3, 4, 7, 1, 3, 4]
        s = 0.1
        tx = [1 + s, 3 - s]
        ty = [1 + s, 3 - s]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, '\nThe coefficients of the spline')
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
        tx, ty = lut.get_knots()
        for xa, xb in zip(tx[:-1], tx[1:]):
            for ya, yb in zip(ty[:-1], ty[1:]):
                for t in [0.1, 0.5, 0.9]:
                    for s in [0.3, 0.4, 0.7]:
                        xp = xa * (1 - t) + xb * t
                        yp = ya * (1 - s) + yb * s
                        zp = +lut(xa, ya) * (1 - t) * (1 - s) + lut(xb, ya) * t * (1 - s) + lut(xa, yb) * (1 - t) * s + lut(xb, yb) * t * s
                        assert_almost_equal(lut(xp, yp), zp)

    def test_integral(self):
        x = [1, 1, 1, 2, 2, 2, 8, 8, 8]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = array([0, 7, 8, 3, 4, 7, 1, 3, 4])
        s = 0.1
        tx = [1 + s, 3 - s]
        ty = [1 + s, 3 - s]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, '\nThe coefficients of the spline')
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
            assert_equal(len(r), 1)
        tx, ty = lut.get_knots()
        tz = lut(tx, ty)
        trpz = 0.25 * (diff(tx)[:, None] * diff(ty)[None, :] * (tz[:-1, :-1] + tz[1:, :-1] + tz[:-1, 1:] + tz[1:, 1:])).sum()
        assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz)

    def test_empty_input(self):
        x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        z = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        s = 0.1
        tx = [1 + s, 3 - s]
        ty = [1 + s, 3 - s]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, '\nThe coefficients of the spline')
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
            assert_equal(len(r), 1)
        assert_array_equal(lut([], []), np.zeros((0, 0)))
        assert_array_equal(lut([], [], grid=False), np.zeros((0,)))

    def test_invalid_input(self):
        s = 0.1
        tx = [1 + s, 3 - s]
        ty = [1 + s, 3 - s]
        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0, num=10)
            LSQBivariateSpline(x, y, z, tx, ty)
        assert 'x, y, and z should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0)
            w = np.linspace(1.0, 10.0, num=20)
            LSQBivariateSpline(x, y, z, tx, ty, w=w)
        assert 'x, y, z, and w should have a same length' in str(info.value)
        with assert_raises(ValueError) as info:
            w = np.linspace(-1.0, 10.0)
            LSQBivariateSpline(x, y, z, tx, ty, w=w)
        assert 'w should be positive' in str(info.value)
        with assert_raises(ValueError) as info:
            bbox = (-100, 100, -100)
            LSQBivariateSpline(x, y, z, tx, ty, bbox=bbox)
        assert 'bbox shape should be (4,)' in str(info.value)
        with assert_raises(ValueError) as info:
            LSQBivariateSpline(x, y, z, tx, ty, kx=10, ky=10)
        assert 'The length of x, y and z should be at least (kx+1) * (ky+1)' in str(info.value)
        with assert_raises(ValueError) as exc_info:
            LSQBivariateSpline(x, y, z, tx, ty, eps=0.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            LSQBivariateSpline(x, y, z, tx, ty, eps=1.0)
        assert 'eps should be between (0, 1)' in str(exc_info.value)

    def test_array_like_input(self):
        s = 0.1
        tx = np.array([1 + s, 3 - s])
        ty = np.array([1 + s, 3 - s])
        x = np.linspace(1.0, 10.0)
        y = np.linspace(1.0, 10.0)
        z = np.linspace(1.0, 10.0)
        w = np.linspace(1.0, 10.0)
        bbox = np.array([1.0, 10.0, 1.0, 10.0])
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, '\nThe coefficients of the spline')
            spl1 = LSQBivariateSpline(x, y, z, tx, ty, w=w, bbox=bbox)
            spl2 = LSQBivariateSpline(x.tolist(), y.tolist(), z.tolist(), tx.tolist(), ty.tolist(), w=w.tolist(), bbox=bbox)
            assert_allclose(spl1(2.0, 2.0), spl2(2.0, 2.0))
            assert_equal(len(r), 2)

    def test_unequal_length_of_knots(self):
        """Test for the case when the input knot-location arrays in x and y are
        of different lengths.
        """
        x, y = np.mgrid[0:100, 0:100]
        x = x.ravel()
        y = y.ravel()
        z = 3.0 * np.ones_like(x)
        tx = np.linspace(0.1, 98.0, 29)
        ty = np.linspace(0.1, 98.0, 33)
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, '\nThe coefficients of the spline')
            lut = LSQBivariateSpline(x, y, z, tx, ty)
            assert_equal(len(r), 1)
        assert_almost_equal(lut(x, y, grid=False), z)