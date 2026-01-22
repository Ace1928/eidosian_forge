from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
class TestDatasetRolling:

    @pytest.mark.parametrize('funcname, argument', [('reduce', (np.mean,)), ('mean', ()), ('construct', ('window_dim',)), ('count', ())])
    def test_rolling_keep_attrs(self, funcname, argument) -> None:
        global_attrs = {'units': 'test', 'long_name': 'testing'}
        da_attrs = {'da_attr': 'test'}
        da_not_rolled_attrs = {'da_not_rolled_attr': 'test'}
        data = np.linspace(10, 15, 100)
        coords = np.linspace(1, 10, 100)
        ds = Dataset(data_vars={'da': ('coord', data), 'da_not_rolled': ('no_coord', data)}, coords={'coord': coords}, attrs=global_attrs)
        ds.da.attrs = da_attrs
        ds.da_not_rolled.attrs = da_not_rolled_attrs
        func = getattr(ds.rolling(dim={'coord': 5}), funcname)
        result = func(*argument)
        assert result.attrs == global_attrs
        assert result.da.attrs == da_attrs
        assert result.da_not_rolled.attrs == da_not_rolled_attrs
        assert result.da.name == 'da'
        assert result.da_not_rolled.name == 'da_not_rolled'
        func = getattr(ds.rolling(dim={'coord': 5}), funcname)
        result = func(*argument, keep_attrs=False)
        assert result.attrs == {}
        assert result.da.attrs == {}
        assert result.da_not_rolled.attrs == {}
        assert result.da.name == 'da'
        assert result.da_not_rolled.name == 'da_not_rolled'
        func = getattr(ds.rolling(dim={'coord': 5}), funcname)
        with set_options(keep_attrs=False):
            result = func(*argument)
        assert result.attrs == {}
        assert result.da.attrs == {}
        assert result.da_not_rolled.attrs == {}
        assert result.da.name == 'da'
        assert result.da_not_rolled.name == 'da_not_rolled'
        func = getattr(ds.rolling(dim={'coord': 5}), funcname)
        with set_options(keep_attrs=False):
            result = func(*argument, keep_attrs=True)
        assert result.attrs == global_attrs
        assert result.da.attrs == da_attrs
        assert result.da_not_rolled.attrs == da_not_rolled_attrs
        assert result.da.name == 'da'
        assert result.da_not_rolled.name == 'da_not_rolled'
        func = getattr(ds.rolling(dim={'coord': 5}), funcname)
        with set_options(keep_attrs=True):
            result = func(*argument, keep_attrs=False)
        assert result.attrs == {}
        assert result.da.attrs == {}
        assert result.da_not_rolled.attrs == {}
        assert result.da.name == 'da'
        assert result.da_not_rolled.name == 'da_not_rolled'

    def test_rolling_properties(self, ds) -> None:
        with pytest.raises(ValueError, match='window must be > 0'):
            ds.rolling(time=-2)
        with pytest.raises(ValueError, match='min_periods must be greater than zero'):
            ds.rolling(time=2, min_periods=0)
        with pytest.raises(KeyError, match='time2'):
            ds.rolling(time2=2)
        with pytest.raises(KeyError, match="\\('foo',\\) not found in Dataset dimensions"):
            ds.rolling(foo=2)

    @pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'var', 'min', 'max', 'median'))
    @pytest.mark.parametrize('center', (True, False, None))
    @pytest.mark.parametrize('min_periods', (1, None))
    @pytest.mark.parametrize('key', ('z1', 'z2'))
    @pytest.mark.parametrize('backend', ['numpy'], indirect=True)
    def test_rolling_wrapped_bottleneck(self, ds, name, center, min_periods, key, compute_backend) -> None:
        bn = pytest.importorskip('bottleneck', minversion='1.1')
        rolling_obj = ds.rolling(time=7, min_periods=min_periods)
        func_name = f'move_{name}'
        actual = getattr(rolling_obj, name)()
        if key == 'z1':
            expected = ds[key]
        elif key == 'z2':
            expected = getattr(bn, func_name)(ds[key].values, window=7, axis=0, min_count=min_periods)
        else:
            raise ValueError
        np.testing.assert_allclose(actual[key].values, expected)
        rolling_obj = ds.rolling(time=7, center=center)
        actual = getattr(rolling_obj, name)()['time']
        assert_allclose(actual, ds['time'])

    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    def test_rolling_pandas_compat(self, center, window, min_periods) -> None:
        df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 'time': np.linspace(0, 1, 20)})
        ds = Dataset.from_dataframe(df)
        if min_periods is not None and window < min_periods:
            min_periods = window
        df_rolling = df.rolling(window, center=center, min_periods=min_periods).mean()
        ds_rolling = ds.rolling(index=window, center=center, min_periods=min_periods).mean()
        np.testing.assert_allclose(df_rolling['x'].values, ds_rolling['x'].values)
        np.testing.assert_allclose(df_rolling.index, ds_rolling['index'])

    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    def test_rolling_construct(self, center: bool, window: int) -> None:
        df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 'time': np.linspace(0, 1, 20)})
        ds = Dataset.from_dataframe(df)
        df_rolling = df.rolling(window, center=center, min_periods=1).mean()
        ds_rolling = ds.rolling(index=window, center=center)
        ds_rolling_mean = ds_rolling.construct('window').mean('window')
        np.testing.assert_allclose(df_rolling['x'].values, ds_rolling_mean['x'].values)
        np.testing.assert_allclose(df_rolling.index, ds_rolling_mean['index'])
        ds_rolling_mean = ds_rolling.construct('window', stride=2, fill_value=0.0).mean('window')
        assert (ds_rolling_mean.isnull().sum() == 0).to_dataarray(dim='vars').all()
        assert (ds_rolling_mean['x'] == 0.0).sum() >= 0

    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    def test_rolling_construct_stride(self, center: bool, window: int) -> None:
        df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 'time': np.linspace(0, 1, 20)})
        ds = Dataset.from_dataframe(df)
        df_rolling_mean = df.rolling(window, center=center, min_periods=1).mean()
        ds_rolling = ds.rolling(index=window, center=center)
        ds_rolling_mean = ds_rolling.construct('w', stride=2).mean('w')
        np.testing.assert_allclose(df_rolling_mean['x'][::2].values, ds_rolling_mean['x'].values)
        np.testing.assert_allclose(df_rolling_mean.index[::2], ds_rolling_mean['index'])
        ds2 = ds.drop_vars('index')
        ds2_rolling = ds2.rolling(index=window, center=center)
        ds2_rolling_mean = ds2_rolling.construct('w', stride=2).mean('w')
        np.testing.assert_allclose(df_rolling_mean['x'][::2].values, ds2_rolling_mean['x'].values)
        ds3 = xr.Dataset({'x': ('t', range(20)), 'x2': ('y', range(5))}, {'t': range(20), 'y': ('y', range(5)), 't2': ('t', range(20)), 'y2': ('y', range(5)), 'yt': (['t', 'y'], np.ones((20, 5)))})
        ds3_rolling = ds3.rolling(t=window, center=center)
        ds3_rolling_mean = ds3_rolling.construct('w', stride=2).mean('w')
        for coord in ds3.coords:
            assert coord in ds3_rolling_mean.coords

    @pytest.mark.slow
    @pytest.mark.parametrize('ds', (1, 2), indirect=True)
    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    @pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'var', 'min', 'max', 'median'))
    def test_rolling_reduce(self, ds, center, min_periods, window, name) -> None:
        if min_periods is not None and window < min_periods:
            min_periods = window
        if name == 'std' and window == 1:
            pytest.skip('std with window == 1 is unstable in bottleneck')
        rolling_obj = ds.rolling(time=window, center=center, min_periods=min_periods)
        actual = rolling_obj.reduce(getattr(np, 'nan%s' % name))
        expected = getattr(rolling_obj, name)()
        assert_allclose(actual, expected)
        assert ds.sizes == actual.sizes
        assert list(ds.data_vars.keys()) == list(actual.data_vars.keys())
        for key, src_var in ds.data_vars.items():
            assert src_var.dims == actual[key].dims

    @pytest.mark.parametrize('ds', (2,), indirect=True)
    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1))
    @pytest.mark.parametrize('name', ('sum', 'max'))
    @pytest.mark.parametrize('dask', (True, False))
    def test_ndrolling_reduce(self, ds, center, min_periods, name, dask) -> None:
        if dask and has_dask:
            ds = ds.chunk({'x': 4})
        rolling_obj = ds.rolling(time=4, x=3, center=center, min_periods=min_periods)
        actual = getattr(rolling_obj, name)()
        expected = getattr(getattr(ds.rolling(time=4, center=center, min_periods=min_periods), name)().rolling(x=3, center=center, min_periods=min_periods), name)()
        assert_allclose(actual, expected)
        assert actual.sizes == expected.sizes
        expected = getattr(getattr(ds.rolling(x=3, center=center, min_periods=min_periods), name)().rolling(time=4, center=center, min_periods=min_periods), name)()
        assert_allclose(actual, expected)
        assert actual.sizes == expected.sizes

    @pytest.mark.parametrize('center', (True, False, (True, False)))
    @pytest.mark.parametrize('fill_value', (np.nan, 0.0))
    @pytest.mark.parametrize('dask', (True, False))
    def test_ndrolling_construct(self, center, fill_value, dask) -> None:
        da = DataArray(np.arange(5 * 6 * 7).reshape(5, 6, 7).astype(float), dims=['x', 'y', 'z'], coords={'x': ['a', 'b', 'c', 'd', 'e'], 'y': np.arange(6)})
        ds = xr.Dataset({'da': da})
        if dask and has_dask:
            ds = ds.chunk({'x': 4})
        actual = ds.rolling(x=3, z=2, center=center).construct(x='x1', z='z1', fill_value=fill_value)
        if not isinstance(center, tuple):
            center = (center, center)
        expected = ds.rolling(x=3, center=center[0]).construct(x='x1', fill_value=fill_value).rolling(z=2, center=center[1]).construct(z='z1', fill_value=fill_value)
        assert_allclose(actual, expected)

    @requires_dask
    @pytest.mark.filterwarnings('error')
    @pytest.mark.parametrize('ds', (2,), indirect=True)
    @pytest.mark.parametrize('name', ('mean', 'max'))
    def test_raise_no_warning_dask_rolling_assert_close(self, ds, name) -> None:
        """
        This is a puzzle — I can't easily find the source of the warning. It
        requires `assert_allclose` to be run, for the `ds` param to be 2, and is
        different for `mean` and `max`. `sum` raises no warning.
        """
        ds = ds.chunk({'x': 4})
        rolling_obj = ds.rolling(time=4, x=3)
        actual = getattr(rolling_obj, name)()
        expected = getattr(getattr(ds.rolling(time=4), name)().rolling(x=3), name)()
        assert_allclose(actual, expected)