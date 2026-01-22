from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
class TestConcatDataArray:

    def test_concat(self) -> None:
        ds = Dataset({'foo': (['x', 'y'], np.random.random((2, 3))), 'bar': (['x', 'y'], np.random.random((2, 3)))}, {'x': [0, 1]})
        foo = ds['foo']
        bar = ds['bar']
        expected = DataArray(np.array([foo.values, bar.values]), dims=['w', 'x', 'y'], coords={'x': [0, 1]})
        actual = concat([foo, bar], 'w')
        assert_equal(expected, actual)
        grouped = [g.squeeze() for _, g in foo.groupby('x', squeeze=False)]
        stacked = concat(grouped, ds['x'])
        assert_identical(foo, stacked)
        stacked = concat(grouped, pd.Index(ds['x'], name='x'))
        assert_identical(foo, stacked)
        actual2 = concat([foo[0], foo[1]], pd.Index([0, 1])).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        assert_identical(expected, actual2)
        actual3 = concat([foo[0], foo[1]], [0, 1]).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        assert_identical(expected, actual3)
        with pytest.raises(ValueError, match='not identical'):
            concat([foo, bar], dim='w', compat='identical')
        with pytest.raises(ValueError, match='not a valid argument'):
            concat([foo, bar], dim='w', data_vars='minimal')

    def test_concat_encoding(self) -> None:
        ds = Dataset({'foo': (['x', 'y'], np.random.random((2, 3))), 'bar': (['x', 'y'], np.random.random((2, 3)))}, {'x': [0, 1]})
        foo = ds['foo']
        foo.encoding = {'complevel': 5}
        ds.encoding = {'unlimited_dims': 'x'}
        assert concat([foo, foo], dim='x').encoding == foo.encoding
        assert concat([ds, ds], dim='x').encoding == ds.encoding

    @requires_dask
    def test_concat_lazy(self) -> None:
        import dask.array as da
        arrays = [DataArray(da.from_array(InaccessibleArray(np.zeros((3, 3))), 3), dims=['x', 'y']) for _ in range(2)]
        combined = concat(arrays, dim='z')
        assert combined.shape == (2, 3, 3)
        assert combined.dims == ('z', 'x', 'y')

    @pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0])
    def test_concat_fill_value(self, fill_value) -> None:
        foo = DataArray([1, 2], coords=[('x', [1, 2])])
        bar = DataArray([1, 2], coords=[('x', [1, 3])])
        if fill_value == dtypes.NA:
            fill_value = np.nan
        expected = DataArray([[1, 2, fill_value], [1, fill_value, 2]], dims=['y', 'x'], coords={'x': [1, 2, 3]})
        actual = concat((foo, bar), dim='y', fill_value=fill_value)
        assert_identical(actual, expected)

    def test_concat_join_kwarg(self) -> None:
        ds1 = Dataset({'a': (('x', 'y'), [[0]])}, coords={'x': [0], 'y': [0]}).to_dataarray()
        ds2 = Dataset({'a': (('x', 'y'), [[0]])}, coords={'x': [1], 'y': [0.0001]}).to_dataarray()
        expected: dict[JoinOptions, Any] = {}
        expected['outer'] = Dataset({'a': (('x', 'y'), [[0, np.nan], [np.nan, 0]])}, {'x': [0, 1], 'y': [0, 0.0001]})
        expected['inner'] = Dataset({'a': (('x', 'y'), [[], []])}, {'x': [0, 1], 'y': []})
        expected['left'] = Dataset({'a': (('x', 'y'), np.array([0, np.nan], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0]})
        expected['right'] = Dataset({'a': (('x', 'y'), np.array([np.nan, 0], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0.0001]})
        expected['override'] = Dataset({'a': (('x', 'y'), np.array([0, 0], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0]})
        with pytest.raises(ValueError, match="cannot align.*exact.*dimensions.*'y'"):
            actual = concat([ds1, ds2], join='exact', dim='x')
        for join in expected:
            actual = concat([ds1, ds2], join=join, dim='x')
            assert_equal(actual, expected[join].to_dataarray())

    def test_concat_combine_attrs_kwarg(self) -> None:
        da1 = DataArray([0], coords=[('x', [0])], attrs={'b': 42})
        da2 = DataArray([0], coords=[('x', [1])], attrs={'b': 42, 'c': 43})
        expected: dict[CombineAttrsOptions, Any] = {}
        expected['drop'] = DataArray([0, 0], coords=[('x', [0, 1])])
        expected['no_conflicts'] = DataArray([0, 0], coords=[('x', [0, 1])], attrs={'b': 42, 'c': 43})
        expected['override'] = DataArray([0, 0], coords=[('x', [0, 1])], attrs={'b': 42})
        with pytest.raises(ValueError, match="combine_attrs='identical'"):
            actual = concat([da1, da2], dim='x', combine_attrs='identical')
        with pytest.raises(ValueError, match="combine_attrs='no_conflicts'"):
            da3 = da2.copy(deep=True)
            da3.attrs['b'] = 44
            actual = concat([da1, da3], dim='x', combine_attrs='no_conflicts')
        for combine_attrs in expected:
            actual = concat([da1, da2], dim='x', combine_attrs=combine_attrs)
            assert_identical(actual, expected[combine_attrs])

    @pytest.mark.parametrize('dtype', [str, bytes])
    @pytest.mark.parametrize('dim', ['x1', 'x2'])
    def test_concat_str_dtype(self, dtype, dim) -> None:
        data = np.arange(4).reshape([2, 2])
        da1 = DataArray(data=data, dims=['x1', 'x2'], coords={'x1': [0, 1], 'x2': np.array(['a', 'b'], dtype=dtype)})
        da2 = DataArray(data=data, dims=['x1', 'x2'], coords={'x1': np.array([1, 2]), 'x2': np.array(['c', 'd'], dtype=dtype)})
        actual = concat([da1, da2], dim=dim)
        assert np.issubdtype(actual.x2.dtype, dtype)

    def test_concat_coord_name(self) -> None:
        da = DataArray([0], dims='a')
        da_concat = concat([da, da], dim=DataArray([0, 1], dims='b'))
        assert list(da_concat.coords) == ['b']
        da_concat_std = concat([da, da], dim=DataArray([0, 1]))
        assert list(da_concat_std.coords) == ['dim_0']