import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestEmptyFrameSetitemExpansion:

    def test_empty_frame_setitem_index_name_retained(self):
        df = DataFrame({}, index=pd.RangeIndex(0, name='df_index'))
        series = Series(1.23, index=pd.RangeIndex(4, name='series_index'))
        df['series'] = series
        expected = DataFrame({'series': [1.23] * 4}, index=pd.RangeIndex(4, name='df_index'), columns=Index(['series'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_empty_frame_setitem_index_name_inherited(self):
        df = DataFrame()
        series = Series(1.23, index=pd.RangeIndex(4, name='series_index'))
        df['series'] = series
        expected = DataFrame({'series': [1.23] * 4}, index=pd.RangeIndex(4, name='series_index'), columns=Index(['series'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_zerolen_series_columns_align(self):
        df = DataFrame(columns=['A', 'B'])
        df.loc[0] = Series(1, index=range(4))
        expected = DataFrame(columns=['A', 'B'], index=[0], dtype=np.float64)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=['A', 'B'])
        df.loc[0] = Series(1, index=['B'])
        exp = DataFrame([[np.nan, 1]], columns=['A', 'B'], index=[0], dtype='float64')
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_zerolen_list_length_must_match_columns(self):
        df = DataFrame(columns=['A', 'B'])
        msg = 'cannot set a row with mismatched columns'
        with pytest.raises(ValueError, match=msg):
            df.loc[0] = [1, 2, 3]
        df = DataFrame(columns=['A', 'B'])
        df.loc[3] = [6, 7]
        exp = DataFrame([[6, 7]], index=[3], columns=['A', 'B'], dtype=np.int64)
        tm.assert_frame_equal(df, exp)

    def test_partial_set_empty_frame(self):
        df = DataFrame()
        msg = 'cannot set a frame with no defined columns'
        with pytest.raises(ValueError, match=msg):
            df.loc[1] = 1
        with pytest.raises(ValueError, match=msg):
            df.loc[1] = Series([1], index=['foo'])
        msg = 'cannot set a frame with no defined index and a scalar'
        with pytest.raises(ValueError, match=msg):
            df.loc[:, 1] = 1

    def test_partial_set_empty_frame2(self):
        expected = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='object'))
        df = DataFrame(index=Index([], dtype='object'))
        df['foo'] = Series([], dtype='object')
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=Index([]))
        df['foo'] = Series(df.index)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=Index([]))
        df['foo'] = df.index
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame3(self):
        expected = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='int64'))
        expected['foo'] = expected['foo'].astype('float64')
        df = DataFrame(index=Index([], dtype='int64'))
        df['foo'] = []
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=Index([], dtype='int64'))
        df['foo'] = Series(np.arange(len(df)), dtype='float64')
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame4(self):
        df = DataFrame(index=Index([], dtype='int64'))
        df['foo'] = range(len(df))
        expected = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='int64'))
        expected['foo'] = expected['foo'].astype('int64')
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame5(self):
        df = DataFrame()
        tm.assert_index_equal(df.columns, pd.RangeIndex(0))
        df2 = DataFrame()
        df2[1] = Series([1], index=['foo'])
        df.loc[:, 1] = Series([1], index=['foo'])
        tm.assert_frame_equal(df, DataFrame([[1]], index=['foo'], columns=[1]))
        tm.assert_frame_equal(df, df2)

    def test_partial_set_empty_frame_no_index(self):
        expected = DataFrame({0: Series(1, index=range(4))}, columns=['A', 'B', 0])
        df = DataFrame(columns=['A', 'B'])
        df[0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=['A', 'B'])
        df.loc[:, 0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_row(self):
        expected = DataFrame(columns=['A', 'B', 'New'], index=Index([], dtype='int64'))
        expected['A'] = expected['A'].astype('int64')
        expected['B'] = expected['B'].astype('float64')
        expected['New'] = expected['New'].astype('float64')
        df = DataFrame({'A': [1, 2, 3], 'B': [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        y['New'] = np.nan
        tm.assert_frame_equal(y, expected)
        expected = DataFrame(columns=['a', 'b', 'c c', 'd'])
        expected['d'] = expected['d'].astype('int64')
        df = DataFrame(columns=['a', 'b', 'c c'])
        df['d'] = 3
        tm.assert_frame_equal(df, expected)
        tm.assert_series_equal(df['c c'], Series(name='c c', dtype=object))
        df = DataFrame({'A': [1, 2, 3], 'B': [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        result = y.reindex(columns=['A', 'B', 'C'])
        expected = DataFrame(columns=['A', 'B', 'C'])
        expected['A'] = expected['A'].astype('int64')
        expected['B'] = expected['B'].astype('float64')
        expected['C'] = expected['C'].astype('float64')
        tm.assert_frame_equal(result, expected)

    def test_partial_set_empty_frame_set_series(self):
        df = DataFrame(Series(dtype=object))
        expected = DataFrame({0: Series(dtype=object)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame(Series(name='foo', dtype=object))
        expected = DataFrame({'foo': Series(dtype=object)})
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_copy_assignment(self):
        df = DataFrame(index=[0])
        df = df.copy()
        df['a'] = 0
        expected = DataFrame(0, index=[0], columns=Index(['a'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string):
        df = DataFrame(columns=['x', 'y'])
        df['x'] = [1, 2]
        expected = DataFrame({'x': [1, 2], 'y': [np.nan, np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)
        df = DataFrame(columns=['x', 'y'])
        df['x'] = ['1', '2']
        expected = DataFrame({'x': Series(['1', '2'], dtype=object if not using_infer_string else 'string[pyarrow_numpy]'), 'y': Series([np.nan, np.nan], dtype=object)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=['x', 'y'])
        df.loc[0, 'x'] = 1
        expected = DataFrame({'x': [1], 'y': [np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)