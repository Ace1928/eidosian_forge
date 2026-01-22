import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
class TestRegularGridInterpolator:

    def _get_sample_4d(self):
        points = [(0.0, 0.5, 1.0)] * 4
        values = np.asarray([0.0, 0.5, 1.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return (points, values)

    def _get_sample_4d_2(self):
        points = [(0.0, 0.5, 1.0)] * 2 + [(0.0, 5.0, 10.0)] * 2
        values = np.asarray([0.0, 0.5, 1.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return (points, values)

    def _get_sample_4d_3(self):
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)] * 4
        values = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return (points, values)

    def _get_sample_4d_4(self):
        points = [(0.0, 1.0)] * 4
        values = np.asarray([0.0, 1.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return (points, values)

    @parametrize_rgi_interp_methods
    def test_list_input(self, method):
        points, values = self._get_sample_4d_3()
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        interp = RegularGridInterpolator(points, values.tolist(), method=method)
        v1 = interp(sample.tolist())
        interp = RegularGridInterpolator(points, values, method=method)
        v2 = interp(sample)
        assert_allclose(v1, v2)

    @pytest.mark.parametrize('method', ['cubic', 'quintic', 'pchip'])
    def test_spline_dim_error(self, method):
        points, values = self._get_sample_4d_4()
        match = 'points in dimension'
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values, method=method)
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match=match):
            interp(sample, method=method)

    @pytest.mark.parametrize('points_values, sample', [(_get_sample_4d, np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])), (_get_sample_4d_2, np.asarray([0.1, 0.1, 10.0, 9.0]))])
    def test_linear_and_slinear_close(self, points_values, sample):
        points, values = points_values(self)
        interp = RegularGridInterpolator(points, values, method='linear')
        v1 = interp(sample)
        interp = RegularGridInterpolator(points, values, method='slinear')
        v2 = interp(sample)
        assert_allclose(v1, v2)

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        points, values = self._get_sample_4d_3()
        values = values - 2j * values
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        interp = RegularGridInterpolator(points, values, method=method)
        rinterp = RegularGridInterpolator(points, values.real, method=method)
        iinterp = RegularGridInterpolator(points, values.imag, method=method)
        v1 = interp(sample)
        v2 = rinterp(sample) + 1j * iinterp(sample)
        assert_allclose(v1, v2)

    def test_cubic_vs_pchip(self):
        x, y = ([1, 2, 3, 4], [1, 2, 3, 4])
        xg, yg = np.meshgrid(x, y, indexing='ij')
        values = (lambda x, y: x ** 4 * y ** 4)(xg, yg)
        cubic = RegularGridInterpolator((x, y), values, method='cubic')
        pchip = RegularGridInterpolator((x, y), values, method='pchip')
        vals_cubic = cubic([1.5, 2])
        vals_pchip = pchip([1.5, 2])
        assert not np.allclose(vals_cubic, vals_pchip, atol=1e-14, rtol=0)

    def test_linear_xi1d(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([0.1, 0.1, 10.0, 9.0])
        wanted = 1001.1
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_xi3d(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        wanted = np.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    @pytest.mark.parametrize('sample, wanted', [(np.asarray([0.1, 0.1, 0.9, 0.9]), 1100.0), (np.asarray([0.1, 0.1, 0.1, 0.1]), 0.0), (np.asarray([0.0, 0.0, 0.0, 0.0]), 0.0), (np.asarray([1.0, 1.0, 1.0, 1.0]), 1111.0), (np.asarray([0.1, 0.4, 0.6, 0.9]), 1055.0)])
    def test_nearest(self, sample, wanted):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method='nearest')
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_edges(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        wanted = np.asarray([0.0, 1111.0])
        assert_array_almost_equal(interp(sample), wanted)

    def test_valid_create(self):
        points = [(0.0, 0.5, 1.0), (0.0, 1.0, 0.5)]
        values = np.asarray([0.0, 0.5, 1.0])
        values0 = values[:, np.newaxis]
        values1 = values[np.newaxis, :]
        values = values0 + values1 * 10
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [((0.0, 0.5, 1.0),), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0.0, 0.5, 0.75, 1.0), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0.0, 0.5, 1.0), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values, method='undefmethod')

    def test_valid_call(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        assert_raises(ValueError, interp, sample, 'undefmethod')
        sample = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert_raises(ValueError, interp, sample)
        sample = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.1]])
        assert_raises(ValueError, interp, sample)

    def test_out_of_bounds_extrap(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False, fill_value=None)
        sample = np.asarray([[-0.1, -0.1, -0.1, -0.1], [1.1, 1.1, 1.1, 1.1], [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([0.0, 1111.0, 11.0, 11.0])
        assert_array_almost_equal(interp(sample, method='nearest'), wanted)
        wanted = np.asarray([-111.1, 1222.1, -11068.0, -1186.9])
        assert_array_almost_equal(interp(sample, method='linear'), wanted)

    def test_out_of_bounds_extrap2(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values, bounds_error=False, fill_value=None)
        sample = np.asarray([[-0.1, -0.1, -0.1, -0.1], [1.1, 1.1, 1.1, 1.1], [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([0.0, 11.0, 11.0, 11.0])
        assert_array_almost_equal(interp(sample, method='nearest'), wanted)
        wanted = np.asarray([-12.1, 133.1, -1069.0, -97.9])
        assert_array_almost_equal(interp(sample, method='linear'), wanted)

    def test_out_of_bounds_fill(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False, fill_value=np.nan)
        sample = np.asarray([[-0.1, -0.1, -0.1, -0.1], [1.1, 1.1, 1.1, 1.1], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([np.nan, np.nan, np.nan])
        assert_array_almost_equal(interp(sample, method='nearest'), wanted)
        assert_array_almost_equal(interp(sample, method='linear'), wanted)
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        wanted = np.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_nearest_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method='nearest')
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values.reshape(-1)
        interp_qhull = NearestNDInterpolator(points_qhull, values_qhull)
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    def test_linear_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values.reshape(-1)
        interp_qhull = LinearNDInterpolator(points_qhull, values_qhull)
        sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    @pytest.mark.parametrize('method', ['nearest', 'linear'])
    def test_duck_typed_values(self, method):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = MyValue((5, 7))
        interp = RegularGridInterpolator((x, y), values, method=method)
        v1 = interp([0.4, 0.7])
        interp = RegularGridInterpolator((x, y), values._v, method=method)
        v2 = interp([0.4, 0.7])
        assert_allclose(v1, v2)

    def test_invalid_fill_value(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        RegularGridInterpolator((x, y), values, fill_value=1)
        assert_raises(ValueError, RegularGridInterpolator, (x, y), values, fill_value=1 + 2j)

    def test_fillvalue_type(self):
        values = np.ones((10, 20, 30), dtype='>f4')
        points = [np.arange(n) for n in values.shape]
        RegularGridInterpolator(points, values)
        RegularGridInterpolator(points, values, fill_value=0.0)

    def test_length_one_axis(self):

        def f(x, y):
            return x + y
        x = np.linspace(1, 1, 1)
        y = np.linspace(1, 10, 10)
        data = f(*np.meshgrid(x, y, indexing='ij', sparse=True))
        interp = RegularGridInterpolator((x, y), data, method='linear', bounds_error=False, fill_value=101)
        assert_allclose(interp(np.array([[1, 1], [1, 5], [1, 10]])), [2, 6, 11], atol=1e-14)
        assert_allclose(interp(np.array([[1, 1.4], [1, 5.3], [1, 10]])), [2.4, 6.3, 11], atol=1e-14)
        assert_allclose(interp(np.array([1.1, 2.4])), interp.fill_value, atol=1e-14)
        interp.fill_value = None
        assert_allclose(interp([[1, 0.3], [1, 11.5]]), [1.3, 12.5], atol=1e-15)
        assert_allclose(interp([[1.5, 0.3], [1.9, 11.5]]), [1.3, 12.5], atol=1e-15)
        interp = RegularGridInterpolator((x, y), data, method='nearest', bounds_error=False, fill_value=None)
        assert_allclose(interp([[1.5, 1.8], [-4, 5.1]]), [3, 6], atol=1e-15)

    @pytest.mark.parametrize('fill_value', [None, np.nan, np.pi])
    @pytest.mark.parametrize('method', ['linear', 'nearest'])
    def test_length_one_axis2(self, fill_value, method):
        options = {'fill_value': fill_value, 'bounds_error': False, 'method': method}
        x = np.linspace(0, 2 * np.pi, 20)
        z = np.sin(x)
        fa = RegularGridInterpolator((x,), z[:], **options)
        fb = RegularGridInterpolator((x, [0]), z[:, None], **options)
        x1a = np.linspace(-1, 2 * np.pi + 1, 100)
        za = fa(x1a)
        y1b = np.zeros(100)
        zb = fb(np.vstack([x1a, y1b]).T)
        assert_allclose(zb, za)
        y1b = np.ones(100)
        zb = fb(np.vstack([x1a, y1b]).T)
        if fill_value is None:
            assert_allclose(zb, za)
        else:
            assert_allclose(zb, fill_value)

    @pytest.mark.parametrize('method', ['nearest', 'linear'])
    def test_nan_x_1d(self, method):
        f = RegularGridInterpolator(([1, 2, 3],), [10, 20, 30], fill_value=1, bounds_error=False, method=method)
        assert np.isnan(f([np.nan]))
        rng = np.random.default_rng(8143215468)
        x = rng.random(size=100) * 4
        i = rng.random(size=100) > 0.5
        x[i] = np.nan
        with np.errstate(invalid='ignore'):
            res = f(x)
        assert_equal(res[i], np.nan)
        assert_equal(res[~i], f(x[~i]))
        x = [1, 2, 3]
        y = [1]
        data = np.ones((3, 1))
        f = RegularGridInterpolator((x, y), data, fill_value=1, bounds_error=False, method=method)
        assert np.isnan(f([np.nan, 1]))
        assert np.isnan(f([1, np.nan]))

    @pytest.mark.parametrize('method', ['nearest', 'linear'])
    def test_nan_x_2d(self, method):
        x, y = (np.array([0, 1, 2]), np.array([1, 3, 7]))

        def f(x, y):
            return x ** 2 + y ** 2
        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        data = f(xg, yg)
        interp = RegularGridInterpolator((x, y), data, method=method, bounds_error=False)
        with np.errstate(invalid='ignore'):
            res = interp([[1.5, np.nan], [1, 1]])
        assert_allclose(res[1], 2, atol=1e-14)
        assert np.isnan(res[0])
        rng = np.random.default_rng(8143215468)
        x = rng.random(size=100) * 4 - 1
        y = rng.random(size=100) * 8
        i1 = rng.random(size=100) > 0.5
        i2 = rng.random(size=100) > 0.5
        i = i1 | i2
        x[i1] = np.nan
        y[i2] = np.nan
        z = np.array([x, y]).T
        with np.errstate(invalid='ignore'):
            res = interp(z)
        assert_equal(res[i], np.nan)
        assert_equal(res[~i], interp(z[~i]))

    @parametrize_rgi_interp_methods
    @pytest.mark.parametrize(('ndims', 'func'), [(2, lambda x, y: 2 * x ** 3 + 3 * y ** 2), (3, lambda x, y, z: 2 * x ** 3 + 3 * y ** 2 - z), (4, lambda x, y, z, a: 2 * x ** 3 + 3 * y ** 2 - z + a), (5, lambda x, y, z, a, b: 2 * x ** 3 + 3 * y ** 2 - z + a * b)])
    def test_descending_points_nd(self, method, ndims, func):
        rng = np.random.default_rng(42)
        sample_low = 1
        sample_high = 5
        test_points = rng.uniform(sample_low, sample_high, size=(2, ndims))
        ascending_points = [np.linspace(sample_low, sample_high, 12) for _ in range(ndims)]
        ascending_values = func(*np.meshgrid(*ascending_points, indexing='ij', sparse=True))
        ascending_interp = RegularGridInterpolator(ascending_points, ascending_values, method=method)
        ascending_result = ascending_interp(test_points)
        descending_points = [xi[::-1] for xi in ascending_points]
        descending_values = func(*np.meshgrid(*descending_points, indexing='ij', sparse=True))
        descending_interp = RegularGridInterpolator(descending_points, descending_values, method=method)
        descending_result = descending_interp(test_points)
        assert_array_equal(ascending_result, descending_result)

    def test_invalid_points_order(self):

        def val_func_2d(x, y):
            return 2 * x ** 3 + 3 * y ** 2
        x = np.array([0.5, 2.0, 0.0, 4.0, 5.5])
        y = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
        points = (x, y)
        values = val_func_2d(*np.meshgrid(*points, indexing='ij', sparse=True))
        match = 'must be strictly ascending or descending'
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values)

    @parametrize_rgi_interp_methods
    def test_fill_value(self, method):
        interp = RegularGridInterpolator([np.arange(6)], np.ones(6), method=method, bounds_error=False)
        assert np.isnan(interp([10]))

    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [(0.0, 5.0, 10.0, 15.0, 20, 25.0)] * 2
        rng = np.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))
        interp = RegularGridInterpolator(points, values, method=method, bounds_error=False)
        v = interp(sample)
        assert_equal(v.shape, (7, 3, 8), err_msg=method)
        vs = []
        for j in range(8):
            interp = RegularGridInterpolator(points, values[..., j], method=method, bounds_error=False)
            vs.append(interp(sample))
        v2 = np.array(vs).transpose(1, 2, 0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    @pytest.mark.parametrize('flip_points', [False, True])
    def test_nonscalar_values_2(self, method, flip_points):
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0), (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]
        if flip_points:
            points = [tuple(reversed(p)) for p in points]
        rng = np.random.default_rng(1234)
        trailing_points = (3, 2)
        values = rng.random((6, 7, 8, 9, *trailing_points))
        sample = rng.random(4)
        interp = RegularGridInterpolator(points, values, method=method, bounds_error=False)
        v = interp(sample)
        assert v.shape == (1, *trailing_points)
        vs = np.empty(values.shape[-2:])
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j], method=method, bounds_error=False)
                vs[i, j] = interp(sample).item()
        v2 = np.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    def test_nonscalar_values_linear_2D(self):
        method = 'linear'
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
        rng = np.random.default_rng(1234)
        trailing_points = (3, 4)
        values = rng.random((6, 7, *trailing_points))
        sample = rng.random(2)
        interp = RegularGridInterpolator(points, values, method=method, bounds_error=False)
        v = interp(sample)
        assert v.shape == (1, *trailing_points)
        vs = np.empty(values.shape[-2:])
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j], method=method, bounds_error=False)
                vs[i, j] = interp(sample).item()
        v2 = np.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
    @pytest.mark.parametrize('xi_dtype', [np.float32, np.float64])
    def test_float32_values(self, dtype, xi_dtype):

        def f(x, y):
            return 2 * x ** 3 + 3 * y ** 2
        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        data = f(xg, yg)
        data = data.astype(dtype)
        interp = RegularGridInterpolator((x, y), data)
        pts = np.array([[2.1, 6.2], [3.3, 5.2]], dtype=xi_dtype)
        assert_allclose(interp(pts), [134.10469388, 153.40069388], atol=1e-07)