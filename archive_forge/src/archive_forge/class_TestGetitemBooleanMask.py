import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
class TestGetitemBooleanMask:

    def test_getitem_bool_mask_categorical_index(self):
        df3 = DataFrame({'A': np.arange(6, dtype='int64')}, index=CategoricalIndex([1, 1, 2, 1, 3, 2], dtype=CategoricalDtype([3, 2, 1], ordered=True), name='B'))
        df4 = DataFrame({'A': np.arange(6, dtype='int64')}, index=CategoricalIndex([1, 1, 2, 1, 3, 2], dtype=CategoricalDtype([3, 2, 1], ordered=False), name='B'))
        result = df3[df3.index == 'a']
        expected = df3.iloc[[]]
        tm.assert_frame_equal(result, expected)
        result = df4[df4.index == 'a']
        expected = df4.iloc[[]]
        tm.assert_frame_equal(result, expected)
        result = df3[df3.index == 1]
        expected = df3.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)
        result = df4[df4.index == 1]
        expected = df4.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)
        result = df3[df3.index < 2]
        expected = df3.iloc[[4]]
        tm.assert_frame_equal(result, expected)
        result = df3[df3.index > 1]
        expected = df3.iloc[[]]
        tm.assert_frame_equal(result, expected)
        msg = 'Unordered Categoricals can only compare equality or not'
        with pytest.raises(TypeError, match=msg):
            df4[df4.index < 2]
        with pytest.raises(TypeError, match=msg):
            df4[df4.index > 1]

    @pytest.mark.parametrize('data1,data2,expected_data', (([[1, 2], [3, 4]], [[0.5, 6], [7, 8]], [[np.nan, 3.0], [np.nan, 4.0], [np.nan, 7.0], [6.0, 8.0]]), ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[np.nan, 3.0], [np.nan, 4.0], [5, 7], [6, 8]])))
    def test_getitem_bool_mask_duplicate_columns_mixed_dtypes(self, data1, data2, expected_data):
        df1 = DataFrame(np.array(data1))
        df2 = DataFrame(np.array(data2))
        df = concat([df1, df2], axis=1)
        result = df[df > 2]
        exdict = {i: np.array(col) for i, col in enumerate(expected_data)}
        expected = DataFrame(exdict).rename(columns={2: 0, 3: 1})
        tm.assert_frame_equal(result, expected)

    @pytest.fixture
    def df_dup_cols(self):
        dups = ['A', 'A', 'C', 'D']
        df = DataFrame(np.arange(12).reshape(3, 4), columns=dups, dtype='float64')
        return df

    def test_getitem_boolean_frame_unaligned_with_duplicate_columns(self, df_dup_cols):
        df = df_dup_cols
        msg = 'cannot reindex on an axis with duplicate labels'
        with pytest.raises(ValueError, match=msg):
            df[df.A > 6]

    def test_getitem_boolean_series_with_duplicate_columns(self, df_dup_cols):
        df = DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'], dtype='float64')
        expected = df[df.C > 6]
        expected.columns = df_dup_cols.columns
        df = df_dup_cols
        result = df[df.C > 6]
        tm.assert_frame_equal(result, expected)

    def test_getitem_boolean_frame_with_duplicate_columns(self, df_dup_cols):
        df = DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'], dtype='float64')
        expected = df[df > 6]
        expected.columns = df_dup_cols.columns
        df = df_dup_cols
        result = df[df > 6]
        tm.assert_frame_equal(result, expected)

    def test_getitem_empty_frame_with_boolean(self):
        df = DataFrame()
        df2 = df[df > 0]
        tm.assert_frame_equal(df, df2)

    def test_getitem_returns_view_when_column_is_unique_in_df(self, using_copy_on_write, warn_copy_on_write):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
        df_orig = df.copy()
        view = df['b']
        with tm.assert_cow_warning(warn_copy_on_write):
            view.loc[:] = 100
        if using_copy_on_write:
            expected = df_orig
        else:
            expected = DataFrame([[1, 2, 100], [4, 5, 100]], columns=['a', 'a', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_getitem_frozenset_unique_in_column(self):
        df = DataFrame([[1, 2, 3, 4]], columns=[frozenset(['KEY']), 'B', 'C', 'C'])
        result = df[frozenset(['KEY'])]
        expected = Series([1], name=frozenset(['KEY']))
        tm.assert_series_equal(result, expected)