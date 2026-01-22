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
class VariableSubclassobjects(NamedArraySubclassobjects, ABC):

    @pytest.fixture
    def target(self, data):
        data = 0.5 * np.arange(10).reshape(2, 5)
        return Variable(['x', 'y'], data)

    def test_getitem_dict(self):
        v = self.cls(['x'], np.random.randn(5))
        actual = v[{'x': 0}]
        expected = v[0]
        assert_identical(expected, actual)

    def test_getitem_1d(self):
        data = np.array([0, 1, 2])
        v = self.cls(['x'], data)
        v_new = v[dict(x=[0, 1])]
        assert v_new.dims == ('x',)
        assert_array_equal(v_new, data[[0, 1]])
        v_new = v[dict(x=slice(None))]
        assert v_new.dims == ('x',)
        assert_array_equal(v_new, data)
        v_new = v[dict(x=Variable('a', [0, 1]))]
        assert v_new.dims == ('a',)
        assert_array_equal(v_new, data[[0, 1]])
        v_new = v[dict(x=1)]
        assert v_new.dims == ()
        assert_array_equal(v_new, data[1])
        v_new = v[slice(None)]
        assert v_new.dims == ('x',)
        assert_array_equal(v_new, data)

    def test_getitem_1d_fancy(self):
        v = self.cls(['x'], [0, 1, 2])
        ind = Variable(('a', 'b'), [[0, 1], [0, 1]])
        v_new = v[ind]
        assert v_new.dims == ('a', 'b')
        expected = np.array(v._data)[([0, 1], [0, 1]), ...]
        assert_array_equal(v_new, expected)
        ind = Variable(('x',), [True, False, True])
        v_new = v[ind]
        assert_identical(v[[0, 2]], v_new)
        v_new = v[[True, False, True]]
        assert_identical(v[[0, 2]], v_new)
        with pytest.raises(IndexError, match='Boolean indexer should'):
            ind = Variable(('a',), [True, False, True])
            v[ind]

    def test_getitem_with_mask(self):
        v = self.cls(['x'], [0, 1, 2])
        assert_identical(v._getitem_with_mask(-1), Variable((), np.nan))
        assert_identical(v._getitem_with_mask([0, -1, 1]), self.cls(['x'], [0, np.nan, 1]))
        assert_identical(v._getitem_with_mask(slice(2)), self.cls(['x'], [0, 1]))
        assert_identical(v._getitem_with_mask([0, -1, 1], fill_value=-99), self.cls(['x'], [0, -99, 1]))

    def test_getitem_with_mask_size_zero(self):
        v = self.cls(['x'], [])
        assert_identical(v._getitem_with_mask(-1), Variable((), np.nan))
        assert_identical(v._getitem_with_mask([-1, -1, -1]), self.cls(['x'], [np.nan, np.nan, np.nan]))

    def test_getitem_with_mask_nd_indexer(self):
        v = self.cls(['x'], [0, 1, 2])
        indexer = Variable(('x', 'y'), [[0, -1], [-1, 2]])
        assert_identical(v._getitem_with_mask(indexer, fill_value=-1), indexer)

    def _assertIndexedLikeNDArray(self, variable, expected_value0, expected_dtype=None):
        """Given a 1-dimensional variable, verify that the variable is indexed
        like a numpy.ndarray.
        """
        assert variable[0].shape == ()
        assert variable[0].ndim == 0
        assert variable[0].size == 1
        assert variable.equals(variable.copy())
        assert variable.identical(variable.copy())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "In the future, 'NAT == x'")
            np.testing.assert_equal(variable.values[0], expected_value0)
            np.testing.assert_equal(variable[0].values, expected_value0)
        if expected_dtype is None:
            assert type(variable.values[0]) == type(expected_value0)
            assert type(variable[0].values) == type(expected_value0)
        elif expected_dtype is not False:
            assert variable.values[0].dtype == expected_dtype
            assert variable[0].values.dtype == expected_dtype

    def test_index_0d_int(self):
        for value, dtype in [(0, np.int_), (np.int32(0), np.int32)]:
            x = self.cls(['x'], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_float(self):
        for value, dtype in [(0.5, float), (np.float32(0.5), np.float32)]:
            x = self.cls(['x'], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_string(self):
        value = 'foo'
        dtype = np.dtype('U3')
        x = self.cls(['x'], [value])
        self._assertIndexedLikeNDArray(x, value, dtype)

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_index_0d_datetime(self):
        d = datetime(2000, 1, 1)
        x = self.cls(['x'], [d])
        self._assertIndexedLikeNDArray(x, np.datetime64(d))
        x = self.cls(['x'], [np.datetime64(d)])
        self._assertIndexedLikeNDArray(x, np.datetime64(d), 'datetime64[ns]')
        x = self.cls(['x'], pd.DatetimeIndex([d]))
        self._assertIndexedLikeNDArray(x, np.datetime64(d), 'datetime64[ns]')

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_index_0d_timedelta64(self):
        td = timedelta(hours=1)
        x = self.cls(['x'], [np.timedelta64(td)])
        self._assertIndexedLikeNDArray(x, np.timedelta64(td), 'timedelta64[ns]')
        x = self.cls(['x'], pd.to_timedelta([td]))
        self._assertIndexedLikeNDArray(x, np.timedelta64(td), 'timedelta64[ns]')

    def test_index_0d_not_a_time(self):
        d = np.datetime64('NaT', 'ns')
        x = self.cls(['x'], [d])
        self._assertIndexedLikeNDArray(x, d)

    def test_index_0d_object(self):

        class HashableItemWrapper:

            def __init__(self, item):
                self.item = item

            def __eq__(self, other):
                return self.item == other.item

            def __hash__(self):
                return hash(self.item)

            def __repr__(self):
                return f'{type(self).__name__}(item={self.item!r})'
        item = HashableItemWrapper((1, 2, 3))
        x = self.cls('x', [item])
        self._assertIndexedLikeNDArray(x, item, expected_dtype=False)

    def test_0d_object_array_with_list(self):
        listarray = np.empty((1,), dtype=object)
        listarray[0] = [1, 2, 3]
        x = self.cls('x', listarray)
        assert_array_equal(x.data, listarray)
        assert_array_equal(x[0].data, listarray.squeeze())
        assert_array_equal(x.squeeze().data, listarray.squeeze())

    def test_index_and_concat_datetime(self):
        date_range = pd.date_range('2011-09-01', periods=10)
        for dates in [date_range, date_range.values, date_range.to_pydatetime()]:
            expected = self.cls('t', dates)
            for times in [[expected[i] for i in range(10)], [expected[i:i + 1] for i in range(10)], [expected[[i]] for i in range(10)]]:
                actual = Variable.concat(times, 't')
                assert expected.dtype == actual.dtype
                assert_array_equal(expected, actual)

    def test_0d_time_data(self):
        x = self.cls('time', pd.date_range('2000-01-01', periods=5))
        expected = np.datetime64('2000-01-01', 'ns')
        assert x[0].values == expected

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_datetime64_conversion(self):
        times = pd.date_range('2000-01-01', periods=3)
        for values, preserve_source in [(times, True), (times.values, True), (times.values.astype('datetime64[s]'), False), (times.to_pydatetime(), False)]:
            v = self.cls(['t'], values)
            assert v.dtype == np.dtype('datetime64[ns]')
            assert_array_equal(v.values, times.values)
            assert v.values.dtype == np.dtype('datetime64[ns]')
            same_source = source_ndarray(v.values) is source_ndarray(values)
            assert preserve_source == same_source

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_timedelta64_conversion(self):
        times = pd.timedelta_range(start=0, periods=3)
        for values, preserve_source in [(times, True), (times.values, True), (times.values.astype('timedelta64[s]'), False), (times.to_pytimedelta(), False)]:
            v = self.cls(['t'], values)
            assert v.dtype == np.dtype('timedelta64[ns]')
            assert_array_equal(v.values, times.values)
            assert v.values.dtype == np.dtype('timedelta64[ns]')
            same_source = source_ndarray(v.values) is source_ndarray(values)
            assert preserve_source == same_source

    def test_object_conversion(self):
        data = np.arange(5).astype(str).astype(object)
        actual = self.cls('x', data)
        assert actual.dtype == data.dtype

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_datetime64_valid_range(self):
        data = np.datetime64('1250-01-01', 'us')
        pderror = pd.errors.OutOfBoundsDatetime
        with pytest.raises(pderror, match='Out of bounds nanosecond'):
            self.cls(['t'], [data])

    @pytest.mark.xfail(reason='pandas issue 36615')
    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_timedelta64_valid_range(self):
        data = np.timedelta64('200000', 'D')
        pderror = pd.errors.OutOfBoundsTimedelta
        with pytest.raises(pderror, match='Out of bounds nanosecond'):
            self.cls(['t'], [data])

    def test_pandas_data(self):
        v = self.cls(['x'], pd.Series([0, 1, 2], index=[3, 2, 1]))
        assert_identical(v, v[[0, 1, 2]])
        v = self.cls(['x'], pd.Index([0, 1, 2]))
        assert v[0].values == v.values[0]

    def test_pandas_period_index(self):
        v = self.cls(['x'], pd.period_range(start='2000', periods=20, freq='D'))
        v = v.load()
        assert v[0] == pd.Period('2000', freq='D')
        assert "Period('2000-01-01', 'D')" in repr(v)

    @pytest.mark.parametrize('dtype', [float, int])
    def test_1d_math(self, dtype: np.typing.DTypeLike) -> None:
        x = np.arange(5, dtype=dtype)
        y = np.ones(5, dtype=dtype)
        v = self.cls(['x'], x)
        base_v = v.to_base_variable()
        assert_identical(base_v, +v)
        assert_identical(base_v, abs(v))
        assert_array_equal((-v).values, -x)
        assert_identical(base_v, v + 0)
        assert_identical(base_v, 0 + v)
        assert_identical(base_v, v * 1)
        if dtype is int:
            assert_identical(base_v, v << 0)
            assert_array_equal(v << 3, x << 3)
            assert_array_equal(v >> 2, x >> 2)
        assert_array_equal((v * x).values, x ** 2)
        assert_array_equal((x * v).values, x ** 2)
        assert_array_equal(v - y, v - 1)
        assert_array_equal(y - v, 1 - v)
        if dtype is int:
            assert_array_equal(v << x, x << x)
            assert_array_equal(v >> x, x >> x)
        v2 = self.cls(['x'], x, {'units': 'meters'})
        with set_options(keep_attrs=False):
            assert_identical(base_v, +v2)
        assert_array_equal(v + v, 2 * v)
        w = self.cls(['x'], y, {'foo': 'bar'})
        assert_identical(v + w, self.cls(['x'], x + y).to_base_variable())
        assert_array_equal((v * w).values, x * y)
        assert_array_equal((v ** 2 * w - 1 + x).values, x ** 2 * y - 1 + x)
        assert dtype == (+v).dtype
        assert dtype == (+v).values.dtype
        assert dtype == (0 + v).dtype
        assert dtype == (0 + v).values.dtype
        assert isinstance(+v, Variable)
        assert not isinstance(+v, IndexVariable)
        assert isinstance(0 + v, Variable)
        assert not isinstance(0 + v, IndexVariable)

    def test_1d_reduce(self):
        x = np.arange(5)
        v = self.cls(['x'], x)
        actual = v.sum()
        expected = Variable((), 10)
        assert_identical(expected, actual)
        assert type(actual) is Variable

    def test_array_interface(self):
        x = np.arange(5)
        v = self.cls(['x'], x)
        assert_array_equal(np.asarray(v), x)
        assert_array_equal(v.astype(float), x.astype(float))
        assert_identical(v.argsort(), v.to_base_variable())
        assert_identical(v.clip(2, 3), self.cls('x', x.clip(2, 3)).to_base_variable())
        assert_identical(np.sin(v), self.cls(['x'], np.sin(x)).to_base_variable())
        assert isinstance(np.sin(v), Variable)
        assert not isinstance(np.sin(v), IndexVariable)

    def example_1d_objects(self):
        for data in [range(3), 0.5 * np.arange(3), 0.5 * np.arange(3, dtype=np.float32), pd.date_range('2000-01-01', periods=3), np.array(['a', 'b', 'c'], dtype=object)]:
            yield (self.cls('x', data), data)

    def test___array__(self):
        for v, data in self.example_1d_objects():
            assert_array_equal(v.values, np.asarray(data))
            assert_array_equal(np.asarray(v), np.asarray(data))
            assert v[0].values == np.asarray(data)[0]
            assert np.asarray(v[0]) == np.asarray(data)[0]

    def test_equals_all_dtypes(self):
        for v, _ in self.example_1d_objects():
            v2 = v.copy()
            assert v.equals(v2)
            assert v.identical(v2)
            assert v.no_conflicts(v2)
            assert v[0].equals(v2[0])
            assert v[0].identical(v2[0])
            assert v[0].no_conflicts(v2[0])
            assert v[:2].equals(v2[:2])
            assert v[:2].identical(v2[:2])
            assert v[:2].no_conflicts(v2[:2])

    def test_eq_all_dtypes(self):
        expected = Variable('x', 3 * [False])
        for v, _ in self.example_1d_objects():
            actual = 'z' == v
            assert_identical(expected, actual)
            actual = ~('z' != v)
            assert_identical(expected, actual)

    def test_encoding_preserved(self):
        expected = self.cls('x', range(3), {'foo': 1}, {'bar': 2})
        for actual in [expected.T, expected[...], expected.squeeze(), expected.isel(x=slice(None)), expected.set_dims({'x': 3}), expected.copy(deep=True), expected.copy(deep=False)]:
            assert_identical(expected.to_base_variable(), actual.to_base_variable())
            assert expected.encoding == actual.encoding

    def test_drop_encoding(self) -> None:
        encoding1 = {'scale_factor': 1}
        v1 = self.cls(['a'], [0, 1, 2], encoding=encoding1)
        assert v1.encoding == encoding1
        v2 = v1.drop_encoding()
        assert v1.encoding == encoding1
        assert v2.encoding == {}
        encoding3 = {'scale_factor': 10}
        v3 = self.cls(['a'], [0, 1, 2], encoding=encoding3)
        assert v3.encoding == encoding3
        v4 = v3.drop_encoding()
        assert v3.encoding == encoding3
        assert v4.encoding == {}

    def test_concat(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        v = self.cls(['a'], x)
        w = self.cls(['a'], y)
        assert_identical(Variable(['b', 'a'], np.array([x, y])), Variable.concat([v, w], 'b'))
        assert_identical(Variable(['b', 'a'], np.array([x, y])), Variable.concat((v, w), 'b'))
        assert_identical(Variable(['b', 'a'], np.array([x, y])), Variable.concat((v, w), 'b'))
        with pytest.raises(ValueError, match='Variable has dimensions'):
            Variable.concat([v, Variable(['c'], y)], 'b')
        actual = Variable.concat([v, w], positions=[np.arange(0, 10, 2), np.arange(1, 10, 2)], dim='a')
        expected = Variable('a', np.array([x, y]).ravel(order='F'))
        assert_identical(expected, actual)
        v = Variable(['time', 'x'], np.random.random((10, 8)))
        assert_identical(v, Variable.concat([v[:5], v[5:]], 'time'))
        assert_identical(v, Variable.concat([v[:5], v[5:6], v[6:]], 'time'))
        assert_identical(v, Variable.concat([v[:1], v[1:]], 'time'))
        assert_identical(v, Variable.concat([v[:, :5], v[:, 5:]], 'x'))
        with pytest.raises(ValueError, match='all input arrays must have'):
            Variable.concat([v[:, 0], v[:, 1:]], 'x')

    def test_concat_attrs(self):
        v = self.cls('a', np.arange(5), {'foo': 'bar'})
        w = self.cls('a', np.ones(5))
        expected = self.cls('a', np.concatenate([np.arange(5), np.ones(5)])).to_base_variable()
        expected.attrs['foo'] = 'bar'
        assert_identical(expected, Variable.concat([v, w], 'a'))

    def test_concat_fixed_len_str(self):
        for kind in ['S', 'U']:
            x = self.cls('animal', np.array(['horse'], dtype=kind))
            y = self.cls('animal', np.array(['aardvark'], dtype=kind))
            actual = Variable.concat([x, y], 'animal')
            expected = Variable('animal', np.array(['horse', 'aardvark'], dtype=kind))
            assert_equal(expected, actual)

    def test_concat_number_strings(self):
        a = self.cls('x', ['0', '1', '2'])
        b = self.cls('x', ['3', '4'])
        actual = Variable.concat([a, b], dim='x')
        expected = Variable('x', np.arange(5).astype(str))
        assert_identical(expected, actual)
        assert actual.dtype.kind == expected.dtype.kind

    def test_concat_mixed_dtypes(self):
        a = self.cls('x', [0, 1])
        b = self.cls('x', ['two'])
        actual = Variable.concat([a, b], dim='x')
        expected = Variable('x', np.array([0, 1, 'two'], dtype=object))
        assert_identical(expected, actual)
        assert actual.dtype == object

    @pytest.mark.parametrize('deep', [True, False])
    @pytest.mark.parametrize('astype', [float, int, str])
    def test_copy(self, deep: bool, astype: type[object]) -> None:
        v = self.cls('x', (0.5 * np.arange(10)).astype(astype), {'foo': 'bar'})
        w = v.copy(deep=deep)
        assert type(v) is type(w)
        assert_identical(v, w)
        assert v.dtype == w.dtype
        if self.cls is Variable:
            if deep:
                assert source_ndarray(v.values) is not source_ndarray(w.values)
            else:
                assert source_ndarray(v.values) is source_ndarray(w.values)
        assert_identical(v, copy(v))

    def test_copy_deep_recursive(self) -> None:
        v = self.cls('x', [0, 1])
        v.attrs['other'] = v
        v.copy(deep=True)
        v2 = self.cls('y', [2, 3])
        v.attrs['other'] = v2
        v2.attrs['other'] = v
        v.copy(deep=True)
        v2.copy(deep=True)

    def test_copy_index(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2], [-1, -2]], names=('one', 'two', 'three'))
        v = self.cls('x', midx)
        for deep in [True, False]:
            w = v.copy(deep=deep)
            assert isinstance(w._data, PandasIndexingAdapter)
            assert isinstance(w.to_index(), pd.MultiIndex)
            assert_array_equal(v._data.array, w._data.array)

    def test_copy_with_data(self) -> None:
        orig = Variable(('x', 'y'), [[1.5, 2.0], [3.1, 4.3]], {'foo': 'bar'})
        new_data = np.array([[2.5, 5.0], [7.1, 43]])
        actual = orig.copy(data=new_data)
        expected = orig.copy()
        expected.data = new_data
        assert_identical(expected, actual)

    def test_copy_with_data_errors(self) -> None:
        orig = Variable(('x', 'y'), [[1.5, 2.0], [3.1, 4.3]], {'foo': 'bar'})
        new_data = [2.5, 5.0]
        with pytest.raises(ValueError, match='must match shape of object'):
            orig.copy(data=new_data)

    def test_copy_index_with_data(self) -> None:
        orig = IndexVariable('x', np.arange(5))
        new_data = np.arange(5, 10)
        actual = orig.copy(data=new_data)
        expected = IndexVariable('x', np.arange(5, 10))
        assert_identical(expected, actual)

    def test_copy_index_with_data_errors(self) -> None:
        orig = IndexVariable('x', np.arange(5))
        new_data = np.arange(5, 20)
        with pytest.raises(ValueError, match='must match shape of object'):
            orig.copy(data=new_data)
        with pytest.raises(ValueError, match='Cannot assign to the .data'):
            orig.data = new_data
        with pytest.raises(ValueError, match='Cannot assign to the .values'):
            orig.values = new_data

    def test_replace(self):
        var = Variable(('x', 'y'), [[1.5, 2.0], [3.1, 4.3]], {'foo': 'bar'})
        result = var._replace()
        assert_identical(result, var)
        new_data = np.arange(4).reshape(2, 2)
        result = var._replace(data=new_data)
        assert_array_equal(result.data, new_data)

    def test_real_and_imag(self):
        v = self.cls('x', np.arange(3) - 1j * np.arange(3), {'foo': 'bar'})
        expected_re = self.cls('x', np.arange(3), {'foo': 'bar'})
        assert_identical(v.real, expected_re)
        expected_im = self.cls('x', -np.arange(3), {'foo': 'bar'})
        assert_identical(v.imag, expected_im)
        expected_abs = self.cls('x', np.sqrt(2 * np.arange(3) ** 2)).to_base_variable()
        assert_allclose(abs(v), expected_abs)

    def test_aggregate_complex(self):
        v = self.cls('x', [1, 2j, np.nan])
        expected = Variable((), 0.5 + 1j)
        assert_allclose(v.mean(), expected)

    def test_pandas_cateogrical_dtype(self):
        data = pd.Categorical(np.arange(10, dtype='int64'))
        v = self.cls('x', data)
        print(v)
        assert v.dtype == 'int64'

    def test_pandas_datetime64_with_tz(self):
        data = pd.date_range(start='2000-01-01', tz=pytz.timezone('America/New_York'), periods=10, freq='1h')
        v = self.cls('x', data)
        print(v)
        if 'America/New_York' in str(data.dtype):
            assert v.dtype == 'object'

    def test_multiindex(self):
        idx = pd.MultiIndex.from_product([list('abc'), [0, 1]])
        v = self.cls('x', idx)
        assert_identical(Variable((), ('a', 0)), v[0])
        assert_identical(v, v[:])

    def test_load(self):
        array = self.cls('x', np.arange(5))
        orig_data = array._data
        copied = array.copy(deep=True)
        if array.chunks is None:
            array.load()
            assert type(array._data) is type(orig_data)
            assert type(copied._data) is type(orig_data)
            assert_identical(array, copied)

    def test_getitem_advanced(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data
        v_new = v[[0, 1], [1, 0]]
        assert v_new.dims == ('x', 'y')
        assert_array_equal(v_new, v_data[[0, 1]][:, [1, 0]])
        v_new = v[[0, 1]]
        assert v_new.dims == ('x', 'y')
        assert_array_equal(v_new, v_data[[0, 1]])
        ind = Variable(['a'], [0, 1])
        v_new = v[dict(x=[0, 1], y=ind)]
        assert v_new.dims == ('x', 'a')
        assert_array_equal(v_new, v_data[[0, 1]][:, [0, 1]])
        v_new = v[dict(x=[True, False], y=[False, True, False])]
        assert v_new.dims == ('x', 'y')
        assert_array_equal(v_new, v_data[0][1])
        ind = Variable((), 2)
        v_new = v[dict(y=ind)]
        expected = v[dict(y=2)]
        assert_array_equal(v_new, expected)
        ind = np.array([True, False])
        with pytest.raises(IndexError, match='Boolean array size 2 is '):
            v[Variable(('a', 'b'), [[0, 1]]), ind]
        ind = Variable(['a'], [True, False, False])
        with pytest.raises(IndexError, match='Boolean indexer should be'):
            v[dict(y=ind)]

    def test_getitem_uint_1d(self):
        v = self.cls(['x'], [0, 1, 2])
        v_data = v.compute().data
        v_new = v[np.array([0])]
        assert_array_equal(v_new, v_data[0])
        v_new = v[np.array([0], dtype='uint64')]
        assert_array_equal(v_new, v_data[0])

    def test_getitem_uint(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data
        v_new = v[np.array([0])]
        assert_array_equal(v_new, v_data[[0], :])
        v_new = v[np.array([0], dtype='uint64')]
        assert_array_equal(v_new, v_data[[0], :])
        v_new = v[np.uint64(0)]
        assert_array_equal(v_new, v_data[0, :])

    def test_getitem_0d_array(self):
        v = self.cls(['x'], [0, 1, 2])
        v_data = v.compute().data
        v_new = v[np.array([0])[0]]
        assert_array_equal(v_new, v_data[0])
        v_new = v[np.array(0)]
        assert_array_equal(v_new, v_data[0])
        v_new = v[Variable((), np.array(0))]
        assert_array_equal(v_new, v_data[0])

    def test_getitem_fancy(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data
        ind = Variable(['a', 'b'], [[0, 1, 1], [1, 1, 0]])
        v_new = v[ind]
        assert v_new.dims == ('a', 'b', 'y')
        assert_array_equal(v_new, v_data[[[0, 1, 1], [1, 1, 0]], :])
        ind = Variable(['x', 'b'], [[0, 1, 1], [1, 1, 0]])
        v_new = v[ind]
        assert v_new.dims == ('x', 'b', 'y')
        assert_array_equal(v_new, v_data[[[0, 1, 1], [1, 1, 0]], :])
        ind = Variable(['a', 'b'], [[0, 1, 2], [2, 1, 0]])
        v_new = v[dict(y=ind)]
        assert v_new.dims == ('x', 'a', 'b')
        assert_array_equal(v_new, v_data[:, ([0, 1, 2], [2, 1, 0])])
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=[1, 0], y=ind)]
        assert v_new.dims == ('x', 'a', 'b')
        assert_array_equal(v_new, v_data[[1, 0]][:, ind])
        ind = Variable(['a'], [0, 1])
        v_new = v[ind, ind]
        assert v_new.dims == ('a',)
        assert_array_equal(v_new, v_data[[0, 1], [0, 1]])
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=0, y=ind)]
        assert v_new.dims == ('a', 'b')
        assert_array_equal(v_new[0], v_data[0][[0, 0]])
        assert_array_equal(v_new[1], v_data[0][[1, 1]])
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=slice(None), y=ind)]
        assert v_new.dims == ('x', 'a', 'b')
        assert_array_equal(v_new, v_data[:, [[0, 0], [1, 1]]])
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=ind, y=slice(None))]
        assert v_new.dims == ('a', 'b', 'y')
        assert_array_equal(v_new, v_data[[[0, 0], [1, 1]], :])
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=ind, y=slice(None, 1))]
        assert v_new.dims == ('a', 'b', 'y')
        assert_array_equal(v_new, v_data[[[0, 0], [1, 1]], slice(None, 1)])
        ind = Variable(['y'], [0, 1])
        v_new = v[ind, :2]
        assert v_new.dims == ('y',)
        assert_array_equal(v_new, v_data[[0, 1], [0, 1]])
        v = self.cls(['x', 'y', 'z'], [[[1, 2, 3], [4, 5, 6]]])
        ind = Variable(['a', 'b'], [[0]])
        v_new = v[ind, :, :]
        expected = Variable(['a', 'b', 'y', 'z'], v.data[np.newaxis, ...])
        assert_identical(v_new, expected)
        v = Variable(['w', 'x', 'y', 'z'], [[[[1, 2, 3], [4, 5, 6]]]])
        ind = Variable(['y'], [0])
        v_new = v[ind, :, 1:2, 2]
        expected = Variable(['y', 'x'], [[6]])
        assert_identical(v_new, expected)
        v = Variable(['x', 'y', 'z'], np.arange(60).reshape(3, 4, 5))
        ind = Variable(['x'], [0, 1, 2])
        v_new = v[:, ind]
        expected = Variable(('x', 'z'), np.zeros((3, 5)))
        expected[0] = v.data[0, 0]
        expected[1] = v.data[1, 1]
        expected[2] = v.data[2, 2]
        assert_identical(v_new, expected)
        v_new = v[:, ind.data]
        assert v_new.shape == (3, 3, 5)

    def test_getitem_error(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        with pytest.raises(IndexError, match='labeled multi-'):
            v[[[0, 1], [1, 2]]]
        ind_x = Variable(['a'], [0, 1, 1])
        ind_y = Variable(['a'], [0, 1])
        with pytest.raises(IndexError, match='Dimensions of indexers '):
            v[ind_x, ind_y]
        ind = Variable(['a', 'b'], [[True, False], [False, True]])
        with pytest.raises(IndexError, match='2-dimensional boolean'):
            v[dict(x=ind)]
        v = Variable(['x', 'y', 'z'], np.arange(60).reshape(3, 4, 5))
        ind = Variable(['x'], [0, 1])
        with pytest.raises(IndexError, match='Dimensions of indexers mis'):
            v[:, ind]

    @pytest.mark.parametrize('mode', ['mean', 'median', 'reflect', 'edge', 'linear_ramp', 'maximum', 'minimum', 'symmetric', 'wrap'])
    @pytest.mark.parametrize('xr_arg, np_arg', _PAD_XR_NP_ARGS)
    @pytest.mark.filterwarnings('ignore:dask.array.pad.+? converts integers to floats.')
    def test_pad(self, mode, xr_arg, np_arg):
        data = np.arange(4 * 3 * 2).reshape(4, 3, 2)
        v = self.cls(['x', 'y', 'z'], data)
        actual = v.pad(mode=mode, **xr_arg)
        expected = np.pad(data, np_arg, mode=mode)
        assert_array_equal(actual, expected)
        assert isinstance(actual._data, type(v._data))

    @pytest.mark.parametrize('xr_arg, np_arg', _PAD_XR_NP_ARGS)
    def test_pad_constant_values(self, xr_arg, np_arg):
        data = np.arange(4 * 3 * 2).reshape(4, 3, 2)
        v = self.cls(['x', 'y', 'z'], data)
        actual = v.pad(**xr_arg)
        expected = np.pad(np.array(duck_array_ops.astype(v.data, float)), np_arg, mode='constant', constant_values=np.nan)
        assert_array_equal(actual, expected)
        assert isinstance(actual._data, type(v._data))
        data = np.full_like(data, False, dtype=bool).reshape(4, 3, 2)
        v = self.cls(['x', 'y', 'z'], data)
        actual = v.pad(mode='constant', constant_values=False, **xr_arg)
        expected = np.pad(np.array(v.data), np_arg, mode='constant', constant_values=False)
        assert_array_equal(actual, expected)

    @pytest.mark.parametrize(['keep_attrs', 'attrs', 'expected'], [pytest.param(None, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, id='default'), pytest.param(False, {'a': 1, 'b': 2}, {}, id='False'), pytest.param(True, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, id='True')])
    def test_pad_keep_attrs(self, keep_attrs, attrs, expected):
        data = np.arange(10, dtype=float)
        v = self.cls(['x'], data, attrs)
        keep_attrs_ = 'default' if keep_attrs is None else keep_attrs
        with set_options(keep_attrs=keep_attrs_):
            actual = v.pad({'x': (1, 1)}, mode='constant', constant_values=np.nan)
            assert actual.attrs == expected
        actual = v.pad({'x': (1, 1)}, mode='constant', constant_values=np.nan, keep_attrs=keep_attrs)
        assert actual.attrs == expected

    @pytest.mark.parametrize('d, w', (('x', 3), ('y', 5)))
    def test_rolling_window(self, d, w):
        v = self.cls(['x', 'y', 'z'], np.arange(40 * 30 * 2).reshape(40, 30, 2))
        v_rolling = v.rolling_window(d, w, d + '_window')
        assert v_rolling.dims == ('x', 'y', 'z', d + '_window')
        assert v_rolling.shape == v.shape + (w,)
        v_rolling = v.rolling_window(d, w, d + '_window', center=True)
        assert v_rolling.dims == ('x', 'y', 'z', d + '_window')
        assert v_rolling.shape == v.shape + (w,)
        v_loaded = v.load().rolling_window(d, w, d + '_window', center=True)
        assert_array_equal(v_rolling, v_loaded)
        if isinstance(v._data, np.ndarray):
            with pytest.raises(ValueError):
                v_loaded[0] = 1.0

    def test_rolling_1d(self):
        x = self.cls('x', np.array([1, 2, 3, 4], dtype=float))
        kwargs = dict(dim='x', window=3, window_dim='xw')
        actual = x.rolling_window(**kwargs, center=True, fill_value=np.nan)
        expected = Variable(('x', 'xw'), np.array([[np.nan, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, np.nan]], dtype=float))
        assert_equal(actual, expected)
        actual = x.rolling_window(**kwargs, center=False, fill_value=0.0)
        expected = self.cls(('x', 'xw'), np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=float))
        assert_equal(actual, expected)
        x = self.cls(('y', 'x'), np.stack([x, x * 1.1]))
        actual = x.rolling_window(**kwargs, center=False, fill_value=0.0)
        expected = self.cls(('y', 'x', 'xw'), np.stack([expected.data, expected.data * 1.1], axis=0))
        assert_equal(actual, expected)

    @pytest.mark.parametrize('center', [[True, True], [False, False]])
    @pytest.mark.parametrize('dims', [('x', 'y'), ('y', 'z'), ('z', 'x')])
    def test_nd_rolling(self, center, dims):
        x = self.cls(('x', 'y', 'z'), np.arange(7 * 6 * 8).reshape(7, 6, 8).astype(float))
        window = [3, 3]
        actual = x.rolling_window(dim=dims, window=window, window_dim=[f'{k}w' for k in dims], center=center, fill_value=np.nan)
        expected = x
        for dim, win, cent in zip(dims, window, center):
            expected = expected.rolling_window(dim=dim, window=win, window_dim=f'{dim}w', center=cent, fill_value=np.nan)
        assert_equal(actual, expected)

    @pytest.mark.parametrize('dim, window, window_dim, center', [('x', [3, 3], 'x_w', True), ('x', 3, ('x_w', 'x_w'), True), ('x', 3, 'x_w', [True, True])])
    def test_rolling_window_errors(self, dim, window, window_dim, center):
        x = self.cls(('x', 'y', 'z'), np.arange(7 * 6 * 8).reshape(7, 6, 8).astype(float))
        with pytest.raises(ValueError):
            x.rolling_window(dim=dim, window=window, window_dim=window_dim, center=center)