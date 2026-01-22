from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
class TestDataArrayResample:

    @pytest.mark.parametrize('use_cftime', [True, False])
    def test_resample(self, use_cftime: bool) -> None:
        if use_cftime and (not has_cftime):
            pytest.skip()
        times = xr.date_range('2000-01-01', freq='6h', periods=10, use_cftime=use_cftime)

        def resample_as_pandas(array, *args, **kwargs):
            array_ = array.copy(deep=True)
            if use_cftime:
                array_['time'] = times.to_datetimeindex()
            result = DataArray.from_series(array_.to_series().resample(*args, **kwargs).mean())
            if use_cftime:
                result = result.convert_calendar(calendar='standard', use_cftime=use_cftime)
            return result
        array = DataArray(np.arange(10), [('time', times)])
        actual = array.resample(time='24h').mean()
        expected = resample_as_pandas(array, '24h')
        assert_identical(expected, actual)
        actual = array.resample(time='24h').reduce(np.mean)
        assert_identical(expected, actual)
        actual = array.resample(time='24h', closed='right').mean()
        expected = resample_as_pandas(array, '24h', closed='right')
        assert_identical(expected, actual)
        with pytest.raises(ValueError, match='index must be monotonic'):
            array[[2, 0, 1]].resample(time='1D')

    @pytest.mark.parametrize('use_cftime', [True, False])
    def test_resample_doctest(self, use_cftime: bool) -> None:
        if use_cftime and (not has_cftime):
            pytest.skip()
        da = xr.DataArray(np.array([1, 2, 3, 1, 2, np.nan]), dims='time', coords=dict(time=('time', xr.date_range('2001-01-01', freq='ME', periods=6, use_cftime=use_cftime)), labels=('time', np.array(['a', 'b', 'c', 'c', 'b', 'a']))))
        actual = da.resample(time='3ME').count()
        expected = DataArray([1, 3, 1], dims='time', coords={'time': xr.date_range('2001-01-01', freq='3ME', periods=3, use_cftime=use_cftime)})
        assert_identical(actual, expected)

    def test_da_resample_func_args(self) -> None:

        def func(arg1, arg2, arg3=0.0):
            return arg1.mean('time') + arg2 + arg3
        times = pd.date_range('2000', periods=3, freq='D')
        da = xr.DataArray([1.0, 1.0, 1.0], coords=[times], dims=['time'])
        expected = xr.DataArray([3.0, 3.0, 3.0], coords=[times], dims=['time'])
        actual = da.resample(time='D').map(func, args=(1.0,), arg3=1.0)
        assert_identical(actual, expected)

    def test_resample_first(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        actual = array.resample(time='6h').first()
        assert_identical(array, actual)
        actual = array.resample(time='1D').first()
        expected = DataArray([0, 4, 8], [('time', times[::4])])
        assert_identical(expected, actual)
        actual = array.resample(time='24h').first()
        expected = DataArray(array.to_series().resample('24h').first())
        assert_identical(expected, actual)
        array = array.astype(float)
        array[:2] = np.nan
        actual = array.resample(time='1D').first()
        expected = DataArray([2, 4, 8], [('time', times[::4])])
        assert_identical(expected, actual)
        actual = array.resample(time='1D').first(skipna=False)
        expected = DataArray([np.nan, 4, 8], [('time', times[::4])])
        assert_identical(expected, actual)
        array = Dataset({'time': times})['time']
        actual = array.resample(time='1D').last()
        expected_times = pd.to_datetime(['2000-01-01T18', '2000-01-02T18', '2000-01-03T06'])
        expected = DataArray(expected_times, [('time', times[::4])], name='time')
        assert_identical(expected, actual)

    def test_resample_bad_resample_dim(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('__resample_dim__', times)])
        with pytest.raises(ValueError, match='Proxy resampling dimension'):
            array.resample(**{'__resample_dim__': '1D'}).first()

    @requires_scipy
    def test_resample_drop_nondim_coords(self) -> None:
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6h', periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        xx, yy = np.meshgrid(xs * 5, ys * 2.5)
        tt = np.arange(len(times), dtype=int)
        array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        xcoord = DataArray(xx.T, {'x': xs, 'y': ys}, ('x', 'y'))
        ycoord = DataArray(yy.T, {'x': xs, 'y': ys}, ('x', 'y'))
        tcoord = DataArray(tt, {'time': times}, ('time',))
        ds = Dataset({'data': array, 'xc': xcoord, 'yc': ycoord, 'tc': tcoord})
        ds = ds.set_coords(['xc', 'yc', 'tc'])
        array = ds['data']
        actual = array.resample(time='12h', restore_coord_dims=True).mean('time')
        assert 'tc' not in actual.coords
        actual = array.resample(time='1h', restore_coord_dims=True).ffill()
        assert 'tc' not in actual.coords
        actual = array.resample(time='1h', restore_coord_dims=True).interpolate('linear')
        assert 'tc' not in actual.coords

    def test_resample_keep_attrs(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        array = DataArray(np.ones(10), [('time', times)])
        array.attrs['meta'] = 'data'
        result = array.resample(time='1D').mean(keep_attrs=True)
        expected = DataArray([1, 1, 1], [('time', times[::4])], attrs=array.attrs)
        assert_identical(result, expected)

    def test_resample_skipna(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        array = DataArray(np.ones(10), [('time', times)])
        array[1] = np.nan
        result = array.resample(time='1D').mean(skipna=False)
        expected = DataArray([np.nan, 1, 1], [('time', times[::4])])
        assert_identical(result, expected)

    def test_upsample(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=5)
        array = DataArray(np.arange(5), [('time', times)])
        actual = array.resample(time='3h').ffill()
        expected = DataArray(array.to_series().resample('3h').ffill())
        assert_identical(expected, actual)
        actual = array.resample(time='3h').bfill()
        expected = DataArray(array.to_series().resample('3h').bfill())
        assert_identical(expected, actual)
        actual = array.resample(time='3h').asfreq()
        expected = DataArray(array.to_series().resample('3h').asfreq())
        assert_identical(expected, actual)
        actual = array.resample(time='3h').pad()
        expected = DataArray(array.to_series().resample('3h').ffill())
        assert_identical(expected, actual)
        rs = array.resample(time='3h')
        actual = rs.nearest()
        new_times = rs.groupers[0].full_index
        expected = DataArray(array.reindex(time=new_times, method='nearest'))
        assert_identical(expected, actual)

    def test_upsample_nd(self) -> None:
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6h', periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        actual = array.resample(time='3h').ffill()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_times = times.to_series().resample('3h').asfreq().index
        expected_data = expected_data[..., :len(expected_times)]
        expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        assert_identical(expected, actual)
        actual = array.resample(time='3h').ffill()
        expected_data = np.repeat(np.flipud(data.T).T, 2, axis=-1)
        expected_data = np.flipud(expected_data.T).T
        expected_times = times.to_series().resample('3h').asfreq().index
        expected_data = expected_data[..., :len(expected_times)]
        expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        assert_identical(expected, actual)
        actual = array.resample(time='3h').asfreq()
        expected_data = np.repeat(data, 2, axis=-1).astype(float)[..., :-1]
        expected_data[..., 1::2] = np.nan
        expected_times = times.to_series().resample('3h').asfreq().index
        expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        assert_identical(expected, actual)
        actual = array.resample(time='3h').pad()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_data[..., 1::2] = expected_data[..., ::2]
        expected_data = expected_data[..., :-1]
        expected_times = times.to_series().resample('3h').asfreq().index
        expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        assert_identical(expected, actual)

    def test_upsample_tolerance(self) -> None:
        times = pd.date_range('2000-01-01', freq='1D', periods=2)
        times_upsampled = pd.date_range('2000-01-01', freq='6h', periods=5)
        array = DataArray(np.arange(2), [('time', times)])
        actual = array.resample(time='6h').ffill(tolerance='12h')
        expected = DataArray([0.0, 0.0, 0.0, np.nan, 1.0], [('time', times_upsampled)])
        assert_identical(expected, actual)
        actual = array.resample(time='6h').bfill(tolerance='12h')
        expected = DataArray([0.0, np.nan, 1.0, 1.0, 1.0], [('time', times_upsampled)])
        assert_identical(expected, actual)
        actual = array.resample(time='6h').nearest(tolerance='6h')
        expected = DataArray([0, 0, np.nan, 1, 1], [('time', times_upsampled)])
        assert_identical(expected, actual)

    @requires_scipy
    def test_upsample_interpolate(self) -> None:
        from scipy.interpolate import interp1d
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6h', periods=5)
        z = np.arange(5) ** 2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        expected_times = times.to_series().resample('1h').asfreq().index
        new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
        kinds: list[InterpOptions] = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        for kind in kinds:
            actual = array.resample(time='1h').interpolate(kind)
            f = interp1d(np.arange(len(times)), data, kind=kind, axis=-1, bounds_error=True, assume_sorted=True)
            expected_data = f(new_times_idx)
            expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
            assert_allclose(expected, actual, rtol=1e-16)

    @requires_scipy
    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_upsample_interpolate_bug_2197(self) -> None:
        dates = pd.date_range('2007-02-01', '2007-03-01', freq='D')
        da = xr.DataArray(np.arange(len(dates)), [('time', dates)])
        result = da.resample(time='ME').interpolate('linear')
        expected_times = np.array([np.datetime64('2007-02-28'), np.datetime64('2007-03-31')])
        expected = xr.DataArray([27.0, np.nan], [('time', expected_times)])
        assert_equal(result, expected)

    @requires_scipy
    def test_upsample_interpolate_regression_1605(self) -> None:
        dates = pd.date_range('2016-01-01', '2016-03-31', freq='1D')
        expected = xr.DataArray(np.random.random((len(dates), 2, 3)), dims=('time', 'x', 'y'), coords={'time': dates})
        actual = expected.resample(time='1D').interpolate('linear')
        assert_allclose(actual, expected, rtol=1e-16)

    @requires_dask
    @requires_scipy
    @pytest.mark.parametrize('chunked_time', [True, False])
    def test_upsample_interpolate_dask(self, chunked_time: bool) -> None:
        from scipy.interpolate import interp1d
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6h', periods=5)
        z = np.arange(5) ** 2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        chunks = {'x': 2, 'y': 1}
        if chunked_time:
            chunks['time'] = 3
        expected_times = times.to_series().resample('1h').asfreq().index
        new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
        kinds: list[InterpOptions] = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        for kind in kinds:
            actual = array.chunk(chunks).resample(time='1h').interpolate(kind)
            actual = actual.compute()
            f = interp1d(np.arange(len(times)), data, kind=kind, axis=-1, bounds_error=True, assume_sorted=True)
            expected_data = f(new_times_idx)
            expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
            assert_allclose(expected, actual, rtol=1e-16)

    @pytest.mark.skipif(has_pandas_version_two, reason='requires pandas < 2.0.0')
    def test_resample_base(self) -> None:
        times = pd.date_range('2000-01-01T02:03:01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        base = 11
        with pytest.warns(FutureWarning, match='the `base` parameter to resample'):
            actual = array.resample(time='24h', base=base).mean()
        expected = DataArray(array.to_series().resample('24h', offset=f'{base}h').mean())
        assert_identical(expected, actual)

    def test_resample_offset(self) -> None:
        times = pd.date_range('2000-01-01T02:03:01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        offset = pd.Timedelta('11h')
        actual = array.resample(time='24h', offset=offset).mean()
        expected = DataArray(array.to_series().resample('24h', offset=offset).mean())
        assert_identical(expected, actual)

    def test_resample_origin(self) -> None:
        times = pd.date_range('2000-01-01T02:03:01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        origin = 'start'
        actual = array.resample(time='24h', origin=origin).mean()
        expected = DataArray(array.to_series().resample('24h', origin=origin).mean())
        assert_identical(expected, actual)

    @pytest.mark.skipif(has_pandas_version_two, reason='requires pandas < 2.0.0')
    @pytest.mark.parametrize('loffset', ['-12H', datetime.timedelta(hours=-12), pd.Timedelta(hours=-12), pd.DateOffset(hours=-12)])
    def test_resample_loffset(self, loffset) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        with pytest.warns(FutureWarning, match='`loffset` parameter'):
            actual = array.resample(time='24h', loffset=loffset).mean()
        series = array.to_series().resample('24h').mean()
        if not isinstance(loffset, pd.DateOffset):
            loffset = pd.Timedelta(loffset)
        series.index = series.index + loffset
        expected = DataArray(series)
        assert_identical(actual, expected)

    def test_resample_invalid_loffset(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        with pytest.warns(FutureWarning, match='Following pandas, the `loffset` parameter'):
            with pytest.raises(ValueError, match='`loffset` must be'):
                array.resample(time='24h', loffset=1).mean()