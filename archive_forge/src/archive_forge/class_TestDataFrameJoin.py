from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
class TestDataFrameJoin:

    def test_join(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        a = frame.loc[frame.index[:5], ['A']]
        b = frame.loc[frame.index[2:], ['B', 'C']]
        joined = a.join(b, how='outer').reindex(frame.index)
        expected = frame.copy().values.copy()
        expected[np.isnan(joined.values)] = np.nan
        expected = DataFrame(expected, index=frame.index, columns=frame.columns)
        assert not np.isnan(joined.values).all()
        tm.assert_frame_equal(joined, expected)

    def test_join_segfault(self):
        df1 = DataFrame({'a': [1, 1], 'b': [1, 2], 'x': [1, 2]})
        df2 = DataFrame({'a': [2, 2], 'b': [1, 2], 'y': [1, 2]})
        df1 = df1.set_index(['a', 'b'])
        df2 = df2.set_index(['a', 'b'])
        for how in ['left', 'right', 'outer']:
            df1.join(df2, how=how)

    def test_join_str_datetime(self):
        str_dates = ['20120209', '20120222']
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]
        A = DataFrame(str_dates, index=range(2), columns=['aa'])
        C = DataFrame([[1, 2], [3, 4]], index=str_dates, columns=dt_dates)
        tst = A.join(C, on='aa')
        assert len(tst.columns) == 3

    def test_join_multiindex_leftright(self):
        df1 = DataFrame([['a', 'x', 0.47178], ['a', 'y', 0.774908], ['a', 'z', 0.563634], ['b', 'x', -0.353756], ['b', 'y', 0.368062], ['b', 'z', -1.72184], ['c', 'x', 1], ['c', 'y', 2], ['c', 'z', 3]], columns=['first', 'second', 'value1']).set_index(['first', 'second'])
        df2 = DataFrame([['a', 10], ['b', 20]], columns=['first', 'value2']).set_index(['first'])
        exp = DataFrame([[0.47178, 10], [0.774908, 10], [0.563634, 10], [-0.353756, 20], [0.368062, 20], [-1.72184, 20], [1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], index=df1.index, columns=['value1', 'value2'])
        tm.assert_frame_equal(df1.join(df2, how='left'), exp)
        tm.assert_frame_equal(df2.join(df1, how='right'), exp[['value2', 'value1']])
        exp_idx = MultiIndex.from_product([['a', 'b'], ['x', 'y', 'z']], names=['first', 'second'])
        exp = DataFrame([[0.47178, 10], [0.774908, 10], [0.563634, 10], [-0.353756, 20], [0.368062, 20], [-1.72184, 20]], index=exp_idx, columns=['value1', 'value2'])
        tm.assert_frame_equal(df1.join(df2, how='right'), exp)
        tm.assert_frame_equal(df2.join(df1, how='left'), exp[['value2', 'value1']])

    def test_join_multiindex_dates(self):
        date = pd.Timestamp(2000, 1, 1).date()
        df1_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
        df1 = DataFrame({'col1': [0]}, index=df1_index)
        df2_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
        df2 = DataFrame({'col2': [0]}, index=df2_index)
        df3_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
        df3 = DataFrame({'col3': [0]}, index=df3_index)
        result = df1.join([df2, df3])
        expected_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
        expected = DataFrame({'col1': [0], 'col2': [0], 'col3': [0]}, index=expected_index)
        tm.assert_equal(result, expected)

    def test_merge_join_different_levels_raises(self):
        df1 = DataFrame(columns=['a', 'b'], data=[[1, 11], [0, 22]])
        columns = MultiIndex.from_tuples([('a', ''), ('c', 'c1')])
        df2 = DataFrame(columns=columns, data=[[1, 33], [0, 44]])
        with pytest.raises(MergeError, match='Not allowed to merge between different levels'):
            pd.merge(df1, df2, on='a')
        with pytest.raises(MergeError, match='Not allowed to merge between different levels'):
            df1.join(df2, on='a')

    def test_frame_join_tzaware(self):
        test1 = DataFrame(np.zeros((6, 3)), index=date_range('2012-11-15 00:00:00', periods=6, freq='100ms', tz='US/Central'))
        test2 = DataFrame(np.zeros((3, 3)), index=date_range('2012-11-15 00:00:00', periods=3, freq='250ms', tz='US/Central'), columns=range(3, 6))
        result = test1.join(test2, how='outer')
        expected = test1.index.union(test2.index)
        tm.assert_index_equal(result.index, expected)
        assert result.index.tz.zone == 'US/Central'