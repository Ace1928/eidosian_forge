import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
class TestRectSphereBivariateSpline:

    def test_defaults(self):
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
        lut = RectSphereBivariateSpline(x, y, z)
        assert_array_almost_equal(lut(x, y), z)

    def test_evaluate(self):
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
        lut = RectSphereBivariateSpline(x, y, z)
        yi = [0.2, 1, 2.3, 2.35, 3.0, 3.99, 5.25]
        xi = [1.5, 0.4, 1.1, 0.45, 0.2345, 1.0, 0.0001]
        zi = lut.ev(xi, yi)
        zi2 = array([lut(xp, yp)[0, 0] for xp, yp in zip(xi, yi)])
        assert_almost_equal(zi, zi2)

    def test_invalid_input(self):
        data = np.dot(np.atleast_2d(90.0 - np.linspace(-80.0, 80.0, 18)).T, np.atleast_2d(180.0 - np.abs(np.linspace(0.0, 350.0, 9)))).T
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(-1, 170, 9) * np.pi / 180.0
            lons = np.linspace(0, 350, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'u should be between (0, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 181, 9) * np.pi / 180.0
            lons = np.linspace(0, 350, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'u should be between (0, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.0
            lons = np.linspace(-181, 10, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'v[0] should be between [-pi, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.0
            lons = np.linspace(-10, 360, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'v[-1] should be v[0] + 2pi or less' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.0
            lons = np.linspace(10, 350, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data, s=-1)
        assert 's should be positive' in str(exc_info.value)

    def test_derivatives_grid(self):
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
        lut = RectSphereBivariateSpline(x, y, z)
        y = linspace(0.02, 2 * pi - 0.02, 7)
        x = linspace(0.02, pi - 0.02, 7)
        assert_allclose(lut(x, y, dtheta=1), _numdiff_2d(lut, x, y, dx=1), rtol=0.0001, atol=0.0001)
        assert_allclose(lut(x, y, dphi=1), _numdiff_2d(lut, x, y, dy=1), rtol=0.0001, atol=0.0001)
        assert_allclose(lut(x, y, dtheta=1, dphi=1), _numdiff_2d(lut, x, y, dx=1, dy=1, eps=1e-06), rtol=0.001, atol=0.001)
        assert_array_equal(lut(x, y, dtheta=1), lut.partial_derivative(1, 0)(x, y))
        assert_array_equal(lut(x, y, dphi=1), lut.partial_derivative(0, 1)(x, y))
        assert_array_equal(lut(x, y, dtheta=1, dphi=1), lut.partial_derivative(1, 1)(x, y))
        assert_array_equal(lut(x, y, dtheta=1, grid=False), lut.partial_derivative(1, 0)(x, y, grid=False))
        assert_array_equal(lut(x, y, dphi=1, grid=False), lut.partial_derivative(0, 1)(x, y, grid=False))
        assert_array_equal(lut(x, y, dtheta=1, dphi=1, grid=False), lut.partial_derivative(1, 1)(x, y, grid=False))

    def test_derivatives(self):
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
        lut = RectSphereBivariateSpline(x, y, z)
        y = linspace(0.02, 2 * pi - 0.02, 7)
        x = linspace(0.02, pi - 0.02, 7)
        assert_equal(lut(x, y, dtheta=1, grid=False).shape, x.shape)
        assert_allclose(lut(x, y, dtheta=1, grid=False), _numdiff_2d(lambda x, y: lut(x, y, grid=False), x, y, dx=1), rtol=0.0001, atol=0.0001)
        assert_allclose(lut(x, y, dphi=1, grid=False), _numdiff_2d(lambda x, y: lut(x, y, grid=False), x, y, dy=1), rtol=0.0001, atol=0.0001)
        assert_allclose(lut(x, y, dtheta=1, dphi=1, grid=False), _numdiff_2d(lambda x, y: lut(x, y, grid=False), x, y, dx=1, dy=1, eps=1e-06), rtol=0.001, atol=0.001)

    def test_invalid_input_2(self):
        data = np.dot(np.atleast_2d(90.0 - np.linspace(-80.0, 80.0, 18)).T, np.atleast_2d(180.0 - np.abs(np.linspace(0.0, 350.0, 9)))).T
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(0, 170, 9) * np.pi / 180.0
            lons = np.linspace(0, 350, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'u should be between (0, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 180, 9) * np.pi / 180.0
            lons = np.linspace(0, 350, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'u should be between (0, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.0
            lons = np.linspace(-181, 10, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'v[0] should be between [-pi, pi)' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.0
            lons = np.linspace(-10, 360, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data)
        assert 'v[-1] should be v[0] + 2pi or less' in str(exc_info.value)
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.0
            lons = np.linspace(10, 350, 18) * np.pi / 180.0
            RectSphereBivariateSpline(lats, lons, data, s=-1)
        assert 's should be positive' in str(exc_info.value)

    def test_array_like_input(self):
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
        spl1 = RectSphereBivariateSpline(x, y, z)
        spl2 = RectSphereBivariateSpline(x.tolist(), y.tolist(), z.tolist())
        assert_array_almost_equal(spl1(x, y), spl2(x, y))

    def test_negative_evaluation(self):
        lats = np.array([25, 30, 35, 40, 45])
        lons = np.array([-90, -85, -80, -75, 70])
        mesh = np.meshgrid(lats, lons)
        data = mesh[0] + mesh[1]
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        interpolator = RectSphereBivariateSpline(lat_r, lon_r, data)
        query_lat = np.radians(np.array([35, 37.5]))
        query_lon = np.radians(np.array([-80, -77.5]))
        data_interp = interpolator(query_lat, query_lon)
        ans = np.array([[-45.0, -42.480862], [-49.0625, -46.54315]])
        assert_array_almost_equal(data_interp, ans)

    def test_pole_continuity_gh_14591(self):
        u = np.arange(1, 10) * np.pi / 10
        v = np.arange(1, 10) * np.pi / 10
        r = np.zeros((9, 9))
        for p in [(True, True), (True, False), (False, False)]:
            RectSphereBivariateSpline(u, v, r, s=0, pole_continuity=p)