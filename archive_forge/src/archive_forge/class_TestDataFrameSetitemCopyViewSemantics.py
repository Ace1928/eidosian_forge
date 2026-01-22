from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
class TestDataFrameSetitemCopyViewSemantics:

    def test_setitem_always_copy(self, float_frame):
        assert 'E' not in float_frame.columns
        s = float_frame['A'].copy()
        float_frame['E'] = s
        float_frame.iloc[5:10, float_frame.columns.get_loc('E')] = np.nan
        assert notna(s[5:10]).all()

    @pytest.mark.parametrize('consolidate', [True, False])
    def test_setitem_partial_column_inplace(self, consolidate, using_array_manager, using_copy_on_write):
        df = DataFrame({'x': [1.1, 2.1, 3.1, 4.1], 'y': [5.1, 6.1, 7.1, 8.1]}, index=[0, 1, 2, 3])
        df.insert(2, 'z', np.nan)
        if not using_array_manager:
            if consolidate:
                df._consolidate_inplace()
                assert len(df._mgr.blocks) == 1
            else:
                assert len(df._mgr.blocks) == 2
        zvals = df['z']._values
        df.loc[2:, 'z'] = 42
        expected = Series([np.nan, np.nan, 42, 42], index=df.index, name='z')
        tm.assert_series_equal(df['z'], expected)
        if not using_copy_on_write:
            tm.assert_numpy_array_equal(zvals, expected.values)
            assert np.shares_memory(zvals, df['z']._values)

    def test_setitem_duplicate_columns_not_inplace(self):
        cols = ['A', 'B'] * 2
        df = DataFrame(0.0, index=[0], columns=cols)
        df_copy = df.copy()
        df_view = df[:]
        df['B'] = (2, 5)
        expected = DataFrame([[0.0, 2, 0.0, 5]], columns=cols)
        tm.assert_frame_equal(df_view, df_copy)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('value', [1, np.array([[1], [1]], dtype='int64'), [[1], [1]]])
    def test_setitem_same_dtype_not_inplace(self, value, using_array_manager):
        cols = ['A', 'B']
        df = DataFrame(0, index=[0, 1], columns=cols)
        df_copy = df.copy()
        df_view = df[:]
        df[['B']] = value
        expected = DataFrame([[0, 1], [0, 1]], columns=cols)
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df_view, df_copy)

    @pytest.mark.parametrize('value', [1.0, np.array([[1.0], [1.0]]), [[1.0], [1.0]]])
    def test_setitem_listlike_key_scalar_value_not_inplace(self, value):
        cols = ['A', 'B']
        df = DataFrame(0, index=[0, 1], columns=cols)
        df_copy = df.copy()
        df_view = df[:]
        df[['B']] = value
        expected = DataFrame([[0, 1.0], [0, 1.0]], columns=cols)
        tm.assert_frame_equal(df_view, df_copy)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', ['a', ['a'], pytest.param([True, False], marks=pytest.mark.xfail(reason='Boolean indexer incorrectly setting inplace', strict=False))])
    @pytest.mark.parametrize('value, set_value', [(1, 5), (1.0, 5.0), (Timestamp('2020-12-31'), Timestamp('2021-12-31')), ('a', 'b')])
    def test_setitem_not_operating_inplace(self, value, set_value, indexer):
        df = DataFrame({'a': value}, index=[0, 1])
        expected = df.copy()
        view = df[:]
        df[indexer] = set_value
        tm.assert_frame_equal(view, expected)

    @td.skip_array_manager_invalid_test
    def test_setitem_column_update_inplace(self, using_copy_on_write, warn_copy_on_write):
        labels = [f'c{i}' for i in range(10)]
        df = DataFrame({col: np.zeros(len(labels)) for col in labels}, index=labels)
        values = df._mgr.blocks[0].values
        with tm.raises_chained_assignment_error():
            for label in df.columns:
                df[label][label] = 1
        if not using_copy_on_write:
            assert np.all(values[np.arange(10), np.arange(10)] == 1)
        else:
            assert np.all(values[np.arange(10), np.arange(10)] == 0)

    def test_setitem_column_frame_as_category(self):
        df = DataFrame([1, 2, 3])
        df['col1'] = DataFrame([1, 2, 3], dtype='category')
        df['col2'] = Series([1, 2, 3], dtype='category')
        expected_types = Series(['int64', 'category', 'category'], index=[0, 'col1', 'col2'], dtype=object)
        tm.assert_series_equal(df.dtypes, expected_types)

    @pytest.mark.parametrize('dtype', ['int64', 'Int64'])
    def test_setitem_iloc_with_numpy_array(self, dtype):
        df = DataFrame({'a': np.ones(3)}, dtype=dtype)
        df.iloc[np.array([0]), np.array([0])] = np.array([[2]])
        expected = DataFrame({'a': [2, 1, 1]}, dtype=dtype)
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_dup_cols_dtype(self):
        df = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=['a', 'b', 'a', 'c'])
        rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=['a', 'a'])
        df['a'] = rhs
        expected = DataFrame([[0, 2, 1.5, 4], [2, 5, 2.5, 7]], columns=['a', 'b', 'a', 'c'])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
        rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=['a', 'a'])
        df['a'] = rhs
        expected = DataFrame([[0, 1.5, 3], [2, 2.5, 6]], columns=['a', 'a', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_frame_setitem_empty_dataframe(self):
        dti = DatetimeIndex(['2000-01-01'], dtype='M8[ns]', name='date')
        df = DataFrame({'date': dti}).set_index('date')
        df = df[0:0].copy()
        df['3010'] = None
        df['2010'] = None
        expected = DataFrame([], columns=['3010', '2010'], index=dti[:0])
        tm.assert_frame_equal(df, expected)