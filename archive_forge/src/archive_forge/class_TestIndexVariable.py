from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
class TestIndexVariable(VariableSubclassobjects):

    def cls(self, *args, **kwargs) -> IndexVariable:
        return IndexVariable(*args, **kwargs)

    def test_init(self):
        with pytest.raises(ValueError, match='must be 1-dimensional'):
            IndexVariable((), 0)

    def test_to_index(self):
        data = 0.5 * np.arange(10)
        v = IndexVariable(['time'], data, {'foo': 'bar'})
        assert pd.Index(data, name='time').identical(v.to_index())

    def test_to_index_multiindex_level(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('one', 'two'))
        ds = Dataset(coords={'x': midx})
        assert ds.one.variable.to_index().equals(midx.get_level_values('one'))

    def test_multiindex_default_level_names(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
        v = IndexVariable(['x'], midx, {'foo': 'bar'})
        assert v.to_index().names == ('x_level_0', 'x_level_1')

    def test_data(self):
        x = IndexVariable('x', np.arange(3.0))
        assert isinstance(x._data, PandasIndexingAdapter)
        assert isinstance(x.data, np.ndarray)
        assert float == x.dtype
        assert_array_equal(np.arange(3), x)
        assert float == x.values.dtype
        with pytest.raises(TypeError, match='cannot be modified'):
            x[:] = 0

    def test_name(self):
        coord = IndexVariable('x', [10.0])
        assert coord.name == 'x'
        with pytest.raises(AttributeError):
            coord.name = 'y'

    def test_level_names(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['level_1', 'level_2'])
        x = IndexVariable('x', midx)
        assert x.level_names == midx.names
        assert IndexVariable('y', [10.0]).level_names is None

    def test_get_level_variable(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['level_1', 'level_2'])
        x = IndexVariable('x', midx)
        level_1 = IndexVariable('x', midx.get_level_values('level_1'))
        assert_identical(x.get_level_variable('level_1'), level_1)
        with pytest.raises(ValueError, match='has no MultiIndex'):
            IndexVariable('y', [10.0]).get_level_variable('level')

    def test_concat_periods(self):
        periods = pd.period_range('2000-01-01', periods=10)
        coords = [IndexVariable('t', periods[:5]), IndexVariable('t', periods[5:])]
        expected = IndexVariable('t', periods)
        actual = IndexVariable.concat(coords, dim='t')
        assert_identical(actual, expected)
        assert isinstance(actual.to_index(), pd.PeriodIndex)
        positions = [list(range(5)), list(range(5, 10))]
        actual = IndexVariable.concat(coords, dim='t', positions=positions)
        assert_identical(actual, expected)
        assert isinstance(actual.to_index(), pd.PeriodIndex)

    def test_concat_multiindex(self):
        idx = pd.MultiIndex.from_product([[0, 1, 2], ['a', 'b']])
        coords = [IndexVariable('x', idx[:2]), IndexVariable('x', idx[2:])]
        expected = IndexVariable('x', idx)
        actual = IndexVariable.concat(coords, dim='x')
        assert_identical(actual, expected)
        assert isinstance(actual.to_index(), pd.MultiIndex)

    @pytest.mark.parametrize('dtype', [str, bytes])
    def test_concat_str_dtype(self, dtype):
        a = IndexVariable('x', np.array(['a'], dtype=dtype))
        b = IndexVariable('x', np.array(['b'], dtype=dtype))
        expected = IndexVariable('x', np.array(['a', 'b'], dtype=dtype))
        actual = IndexVariable.concat([a, b])
        assert actual.identical(expected)
        assert np.issubdtype(actual.dtype, dtype)

    def test_datetime64(self):
        t = np.array([1518418799999986560, 1518418799999996560], dtype='datetime64[ns]')
        v = IndexVariable('t', t)
        assert v[0].data == t[0]

    @pytest.mark.skip
    def test_getitem_error(self):
        super().test_getitem_error()

    @pytest.mark.skip
    def test_getitem_advanced(self):
        super().test_getitem_advanced()

    @pytest.mark.skip
    def test_getitem_fancy(self):
        super().test_getitem_fancy()

    @pytest.mark.skip
    def test_getitem_uint(self):
        super().test_getitem_fancy()

    @pytest.mark.skip
    @pytest.mark.parametrize('mode', ['mean', 'median', 'reflect', 'edge', 'linear_ramp', 'maximum', 'minimum', 'symmetric', 'wrap'])
    @pytest.mark.parametrize('xr_arg, np_arg', _PAD_XR_NP_ARGS)
    def test_pad(self, mode, xr_arg, np_arg):
        super().test_pad(mode, xr_arg, np_arg)

    @pytest.mark.skip
    def test_pad_constant_values(self, xr_arg, np_arg):
        super().test_pad_constant_values(xr_arg, np_arg)

    @pytest.mark.skip
    def test_rolling_window(self):
        super().test_rolling_window()

    @pytest.mark.skip
    def test_rolling_1d(self):
        super().test_rolling_1d()

    @pytest.mark.skip
    def test_nd_rolling(self):
        super().test_nd_rolling()

    @pytest.mark.skip
    def test_rolling_window_errors(self):
        super().test_rolling_window_errors()

    @pytest.mark.skip
    def test_coarsen_2d(self):
        super().test_coarsen_2d()

    def test_to_index_variable_copy(self) -> None:
        a = IndexVariable('x', ['a'])
        b = a.to_index_variable()
        assert a is not b
        b.dims = ('y',)
        assert a.dims == ('x',)