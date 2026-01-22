from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
class TestStackUnstackMultiLevel:

    def test_unstack(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data
        unstacked = ymd.unstack()
        unstacked.unstack()
        ymd.astype(int).unstack()
        ymd.astype(np.int32).unstack()

    @pytest.mark.parametrize('result_rows,result_columns,index_product,expected_row', [([[1, 1, None, None, 30.0, None], [2, 2, None, None, 30.0, None]], ['ix1', 'ix2', 'col1', 'col2', 'col3', 'col4'], 2, [None, None, 30.0, None]), ([[1, 1, None, None, 30.0], [2, 2, None, None, 30.0]], ['ix1', 'ix2', 'col1', 'col2', 'col3'], 2, [None, None, 30.0]), ([[1, 1, None, None, 30.0], [2, None, None, None, 30.0]], ['ix1', 'ix2', 'col1', 'col2', 'col3'], None, [None, None, 30.0])])
    def test_unstack_partial(self, result_rows, result_columns, index_product, expected_row):
        result = DataFrame(result_rows, columns=result_columns).set_index(['ix1', 'ix2'])
        result = result.iloc[1:2].unstack('ix2')
        expected = DataFrame([expected_row], columns=MultiIndex.from_product([result_columns[2:], [index_product]], names=[None, 'ix2']), index=Index([2], name='ix1'))
        tm.assert_frame_equal(result, expected)

    def test_unstack_multiple_no_empty_columns(self):
        index = MultiIndex.from_tuples([(0, 'foo', 0), (0, 'bar', 0), (1, 'baz', 1), (1, 'qux', 1)])
        s = Series(np.random.default_rng(2).standard_normal(4), index=index)
        unstacked = s.unstack([1, 2])
        expected = unstacked.dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack(self, multiindex_year_month_day_dataframe_random_data, future_stack):
        ymd = multiindex_year_month_day_dataframe_random_data
        unstacked = ymd.unstack()
        restacked = unstacked.stack(future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked, ymd)
        unlexsorted = ymd.sort_index(level=2)
        unstacked = unlexsorted.unstack(2)
        restacked = unstacked.stack(future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked.sort_index(level=0), ymd)
        unlexsorted = unlexsorted[::-1]
        unstacked = unlexsorted.unstack(1)
        restacked = unstacked.stack(future_stack=future_stack).swaplevel(1, 2)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked.sort_index(level=0), ymd)
        unlexsorted = unlexsorted.swaplevel(0, 1)
        unstacked = unlexsorted.unstack(0).swaplevel(0, 1, axis=1)
        restacked = unstacked.stack(0, future_stack=future_stack).swaplevel(1, 2)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked.sort_index(level=0), ymd)
        unstacked = ymd.unstack()
        restacked = unstacked.stack(future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked, ymd)
        unstacked = ymd.unstack(1).unstack(1)
        result = unstacked.stack(1, future_stack=future_stack)
        expected = ymd.unstack()
        tm.assert_frame_equal(result, expected)
        result = unstacked.stack(2, future_stack=future_stack)
        expected = ymd.unstack(1)
        tm.assert_frame_equal(result, expected)
        result = unstacked.stack(0, future_stack=future_stack)
        expected = ymd.stack(future_stack=future_stack).unstack(1).unstack(1)
        tm.assert_frame_equal(result, expected)
        unstacked = ymd.unstack(2).loc[:, ::3]
        stacked = unstacked.stack(future_stack=future_stack).stack(future_stack=future_stack)
        ymd_stacked = ymd.stack(future_stack=future_stack)
        if future_stack:
            stacked = stacked.dropna(how='all')
            ymd_stacked = ymd_stacked.dropna(how='all')
        tm.assert_series_equal(stacked, ymd_stacked.reindex(stacked.index))
        result = ymd.unstack(0).stack(-2, future_stack=future_stack)
        expected = ymd.unstack(0).stack(0, future_stack=future_stack)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('idx, columns, exp_idx', [[list('abab'), ['1st', '2nd', '1st'], MultiIndex(levels=[['a', 'b'], ['1st', '2nd']], codes=[np.tile(np.arange(2).repeat(3), 2), np.tile([0, 1, 0], 4)])], [MultiIndex.from_tuples((('a', 2), ('b', 1), ('a', 1), ('b', 2))), ['1st', '2nd', '1st'], MultiIndex(levels=[['a', 'b'], [1, 2], ['1st', '2nd']], codes=[np.tile(np.arange(2).repeat(3), 2), np.repeat([1, 0, 1], [3, 6, 3]), np.tile([0, 1, 0], 4)])]])
    def test_stack_duplicate_index(self, idx, columns, exp_idx, future_stack):
        df = DataFrame(np.arange(12).reshape(4, 3), index=idx, columns=columns)
        if future_stack:
            msg = 'Columns with duplicate values are not supported in stack'
            with pytest.raises(ValueError, match=msg):
                df.stack(future_stack=future_stack)
        else:
            result = df.stack(future_stack=future_stack)
            expected = Series(np.arange(12), index=exp_idx)
            tm.assert_series_equal(result, expected)
            assert result.index.is_unique is False
            li, ri = (result.index, expected.index)
            tm.assert_index_equal(li, ri)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_odd_failure(self, future_stack):
        mi = MultiIndex.from_arrays([['Fri'] * 4 + ['Sat'] * 2 + ['Sun'] * 2 + ['Thu'] * 3, ['Dinner'] * 2 + ['Lunch'] * 2 + ['Dinner'] * 5 + ['Lunch'] * 2, ['No', 'Yes'] * 4 + ['No', 'No', 'Yes']], names=['day', 'time', 'smoker'])
        df = DataFrame({'sum': np.arange(11, dtype='float64'), 'len': np.arange(11, dtype='float64')}, index=mi)
        result = df.unstack(2)
        recons = result.stack(future_stack=future_stack)
        if future_stack:
            recons = recons.dropna(how='all')
        tm.assert_frame_equal(recons, df)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_dtype(self, multiindex_dataframe_random_data, future_stack):
        frame = multiindex_dataframe_random_data
        df = frame.T
        df['foo', 'four'] = 'foo'
        df = df.sort_index(level=1, axis=1)
        stacked = df.stack(future_stack=future_stack)
        result = df['foo'].stack(future_stack=future_stack).sort_index()
        tm.assert_series_equal(stacked['foo'], result, check_names=False)
        assert result.name is None
        assert stacked['bar'].dtype == np.float64

    def test_unstack_bug(self, future_stack):
        df = DataFrame({'state': ['naive', 'naive', 'naive', 'active', 'active', 'active'], 'exp': ['a', 'b', 'b', 'b', 'a', 'a'], 'barcode': [1, 2, 3, 4, 1, 3], 'v': ['hi', 'hi', 'bye', 'bye', 'bye', 'peace'], 'extra': np.arange(6.0)})
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            result = df.groupby(['state', 'exp', 'barcode', 'v']).apply(len)
        unstacked = result.unstack()
        restacked = unstacked.stack(future_stack=future_stack)
        tm.assert_series_equal(restacked, result.reindex(restacked.index).astype(float))

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_preserve_names(self, multiindex_dataframe_random_data, future_stack):
        frame = multiindex_dataframe_random_data
        unstacked = frame.unstack()
        assert unstacked.index.name == 'first'
        assert unstacked.columns.names == ['exp', 'second']
        restacked = unstacked.stack(future_stack=future_stack)
        assert restacked.index.names == frame.index.names

    @pytest.mark.parametrize('method', ['stack', 'unstack'])
    def test_stack_unstack_wrong_level_name(self, method, multiindex_dataframe_random_data, future_stack):
        frame = multiindex_dataframe_random_data
        df = frame.loc['foo']
        kwargs = {'future_stack': future_stack} if method == 'stack' else {}
        with pytest.raises(KeyError, match='does not match index name'):
            getattr(df, method)('mistake', **kwargs)
        if method == 'unstack':
            s = df.iloc[:, 0]
            with pytest.raises(KeyError, match='does not match index name'):
                getattr(s, method)('mistake', **kwargs)

    def test_unstack_level_name(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        result = frame.unstack('second')
        expected = frame.unstack(level=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_level_name(self, multiindex_dataframe_random_data, future_stack):
        frame = multiindex_dataframe_random_data
        unstacked = frame.unstack('second')
        result = unstacked.stack('exp', future_stack=future_stack)
        expected = frame.unstack().stack(0, future_stack=future_stack)
        tm.assert_frame_equal(result, expected)
        result = frame.stack('exp', future_stack=future_stack)
        expected = frame.stack(future_stack=future_stack)
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_multiple(self, multiindex_year_month_day_dataframe_random_data, future_stack):
        ymd = multiindex_year_month_day_dataframe_random_data
        unstacked = ymd.unstack(['year', 'month'])
        expected = ymd.unstack('year').unstack('month')
        tm.assert_frame_equal(unstacked, expected)
        assert unstacked.columns.names == expected.columns.names
        s = ymd['A']
        s_unstacked = s.unstack(['year', 'month'])
        tm.assert_frame_equal(s_unstacked, expected['A'])
        restacked = unstacked.stack(['year', 'month'], future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        restacked = restacked.swaplevel(0, 1).swaplevel(1, 2)
        restacked = restacked.sort_index(level=0)
        tm.assert_frame_equal(restacked, ymd)
        assert restacked.index.names == ymd.index.names
        unstacked = ymd.unstack([1, 2])
        expected = ymd.unstack(1).unstack(1).dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected)
        unstacked = ymd.unstack([2, 1])
        expected = ymd.unstack(2).unstack(1).dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_names_and_numbers(self, multiindex_year_month_day_dataframe_random_data, future_stack):
        ymd = multiindex_year_month_day_dataframe_random_data
        unstacked = ymd.unstack(['year', 'month'])
        with pytest.raises(ValueError, match='level should contain'):
            unstacked.stack([0, 'month'], future_stack=future_stack)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_out_of_bounds(self, multiindex_year_month_day_dataframe_random_data, future_stack):
        ymd = multiindex_year_month_day_dataframe_random_data
        unstacked = ymd.unstack(['year', 'month'])
        with pytest.raises(IndexError, match='Too many levels'):
            unstacked.stack([2, 3], future_stack=future_stack)
        with pytest.raises(IndexError, match='not a valid level number'):
            unstacked.stack([-4, -3], future_stack=future_stack)

    def test_unstack_period_series(self):
        idx1 = pd.PeriodIndex(['2013-01', '2013-01', '2013-02', '2013-02', '2013-03', '2013-03'], freq='M', name='period')
        idx2 = Index(['A', 'B'] * 3, name='str')
        value = [1, 2, 3, 4, 5, 6]
        idx = MultiIndex.from_arrays([idx1, idx2])
        s = Series(value, index=idx)
        result1 = s.unstack()
        result2 = s.unstack(level=1)
        result3 = s.unstack(level=0)
        e_idx = pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period')
        expected = DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6]}, index=e_idx, columns=['A', 'B'])
        expected.columns.name = 'str'
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
        tm.assert_frame_equal(result3, expected.T)
        idx1 = pd.PeriodIndex(['2013-01', '2013-01', '2013-02', '2013-02', '2013-03', '2013-03'], freq='M', name='period1')
        idx2 = pd.PeriodIndex(['2013-12', '2013-11', '2013-10', '2013-09', '2013-08', '2013-07'], freq='M', name='period2')
        idx = MultiIndex.from_arrays([idx1, idx2])
        s = Series(value, index=idx)
        result1 = s.unstack()
        result2 = s.unstack(level=1)
        result3 = s.unstack(level=0)
        e_idx = pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period1')
        e_cols = pd.PeriodIndex(['2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12'], freq='M', name='period2')
        expected = DataFrame([[np.nan, np.nan, np.nan, np.nan, 2, 1], [np.nan, np.nan, 4, 3, np.nan, np.nan], [6, 5, np.nan, np.nan, np.nan, np.nan]], index=e_idx, columns=e_cols)
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
        tm.assert_frame_equal(result3, expected.T)

    def test_unstack_period_frame(self):
        idx1 = pd.PeriodIndex(['2014-01', '2014-02', '2014-02', '2014-02', '2014-01', '2014-01'], freq='M', name='period1')
        idx2 = pd.PeriodIndex(['2013-12', '2013-12', '2014-02', '2013-10', '2013-10', '2014-02'], freq='M', name='period2')
        value = {'A': [1, 2, 3, 4, 5, 6], 'B': [6, 5, 4, 3, 2, 1]}
        idx = MultiIndex.from_arrays([idx1, idx2])
        df = DataFrame(value, index=idx)
        result1 = df.unstack()
        result2 = df.unstack(level=1)
        result3 = df.unstack(level=0)
        e_1 = pd.PeriodIndex(['2014-01', '2014-02'], freq='M', name='period1')
        e_2 = pd.PeriodIndex(['2013-10', '2013-12', '2014-02', '2013-10', '2013-12', '2014-02'], freq='M', name='period2')
        e_cols = MultiIndex.from_arrays(['A A A B B B'.split(), e_2])
        expected = DataFrame([[5, 1, 6, 2, 6, 1], [4, 2, 3, 3, 5, 4]], index=e_1, columns=e_cols)
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
        e_1 = pd.PeriodIndex(['2014-01', '2014-02', '2014-01', '2014-02'], freq='M', name='period1')
        e_2 = pd.PeriodIndex(['2013-10', '2013-12', '2014-02'], freq='M', name='period2')
        e_cols = MultiIndex.from_arrays(['A A B B'.split(), e_1])
        expected = DataFrame([[5, 4, 2, 3], [1, 2, 6, 5], [6, 3, 1, 4]], index=e_2, columns=e_cols)
        tm.assert_frame_equal(result3, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_bug(self, future_stack):
        id_col = [1] * 3 + [2] * 3
        name = ['a'] * 3 + ['b'] * 3
        date = pd.to_datetime(['2013-01-03', '2013-01-04', '2013-01-05'] * 2)
        var1 = np.random.default_rng(2).integers(0, 100, 6)
        df = DataFrame({'ID': id_col, 'NAME': name, 'DATE': date, 'VAR1': var1})
        multi = df.set_index(['DATE', 'ID'])
        multi.columns.name = 'Params'
        unst = multi.unstack('ID')
        msg = re.escape('agg function failed [how->mean,dtype->')
        with pytest.raises(TypeError, match=msg):
            unst.resample('W-THU').mean()
        down = unst.resample('W-THU').mean(numeric_only=True)
        rs = down.stack('ID', future_stack=future_stack)
        xp = unst.loc[:, ['VAR1']].resample('W-THU').mean().stack('ID', future_stack=future_stack)
        xp.columns.name = 'Params'
        tm.assert_frame_equal(rs, xp)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_dropna(self, future_stack):
        df = DataFrame({'A': ['a1', 'a2'], 'B': ['b1', 'b2'], 'C': [1, 1]})
        df = df.set_index(['A', 'B'])
        dropna = False if not future_stack else lib.no_default
        stacked = df.unstack().stack(dropna=dropna, future_stack=future_stack)
        assert len(stacked) > len(stacked.dropna())
        if future_stack:
            with pytest.raises(ValueError, match='dropna must be unspecified'):
                df.unstack().stack(dropna=True, future_stack=future_stack)
        else:
            stacked = df.unstack().stack(dropna=True, future_stack=future_stack)
            tm.assert_frame_equal(stacked, stacked.dropna())

    def test_unstack_multiple_hierarchical(self, future_stack):
        df = DataFrame(index=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], columns=[[0, 0, 1, 1], [0, 1, 0, 1]])
        df.index.names = ['a', 'b', 'c']
        df.columns.names = ['d', 'e']
        df.unstack(['b', 'c'])

    def test_unstack_sparse_keyspace(self):
        NUM_ROWS = 1000
        df = DataFrame({'A': np.random.default_rng(2).integers(100, size=NUM_ROWS), 'B': np.random.default_rng(3).integers(300, size=NUM_ROWS), 'C': np.random.default_rng(4).integers(-7, 7, size=NUM_ROWS), 'D': np.random.default_rng(5).integers(-19, 19, size=NUM_ROWS), 'E': np.random.default_rng(6).integers(3000, size=NUM_ROWS), 'F': np.random.default_rng(7).standard_normal(NUM_ROWS)})
        idf = df.set_index(['A', 'B', 'C', 'D', 'E'])
        idf.unstack('E')

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_unobserved_keys(self, future_stack):
        levels = [[0, 1], [0, 1, 2, 3]]
        codes = [[0, 0, 1, 1], [0, 2, 0, 2]]
        index = MultiIndex(levels, codes)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), index=index)
        result = df.unstack()
        assert len(result.columns) == 4
        recons = result.stack(future_stack=future_stack)
        tm.assert_frame_equal(recons, df)

    @pytest.mark.slow
    def test_unstack_number_of_levels_larger_than_int32(self, monkeypatch):

        class MockUnstacker(reshape_lib._Unstacker):

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                raise Exception("Don't compute final result.")
        with monkeypatch.context() as m:
            m.setattr(reshape_lib, '_Unstacker', MockUnstacker)
            df = DataFrame(np.zeros((2 ** 16, 2)), index=[np.arange(2 ** 16), np.arange(2 ** 16)])
            msg = 'The following operation may generate'
            with tm.assert_produces_warning(PerformanceWarning, match=msg):
                with pytest.raises(Exception, match="Don't compute final result."):
                    df.unstack()

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('levels', itertools.chain.from_iterable((itertools.product(itertools.permutations([0, 1, 2], width), repeat=2) for width in [2, 3])))
    @pytest.mark.parametrize('stack_lev', range(2))
    @pytest.mark.parametrize('sort', [True, False])
    def test_stack_order_with_unsorted_levels(self, levels, stack_lev, sort, future_stack):
        columns = MultiIndex(levels=levels, codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        df = DataFrame(columns=columns, data=[range(4)])
        kwargs = {} if future_stack else {'sort': sort}
        df_stacked = df.stack(stack_lev, future_stack=future_stack, **kwargs)
        for row in df.index:
            for col in df.columns:
                expected = df.loc[row, col]
                result_row = (row, col[stack_lev])
                result_col = col[1 - stack_lev]
                result = df_stacked.loc[result_row, result_col]
                assert result == expected

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_order_with_unsorted_levels_multi_row(self, future_stack):
        mi = MultiIndex(levels=[['A', 'C', 'B'], ['B', 'A', 'C']], codes=[np.repeat(range(3), 3), np.tile(range(3), 3)])
        df = DataFrame(columns=mi, index=range(5), data=np.arange(5 * len(mi)).reshape(5, -1))
        assert all((df.loc[row, col] == df.stack(0, future_stack=future_stack).loc[(row, col[0]), col[1]] for row in df.index for col in df.columns))

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_order_with_unsorted_levels_multi_row_2(self, future_stack):
        levels = ((0, 1), (1, 0))
        stack_lev = 1
        columns = MultiIndex(levels=levels, codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        df = DataFrame(columns=columns, data=[range(4)], index=[1, 0, 2, 3])
        kwargs = {} if future_stack else {'sort': True}
        result = df.stack(stack_lev, future_stack=future_stack, **kwargs)
        expected_index = MultiIndex(levels=[[0, 1, 2, 3], [0, 1]], codes=[[1, 1, 0, 0, 2, 2, 3, 3], [1, 0, 1, 0, 1, 0, 1, 0]])
        expected = DataFrame({0: [0, 1, 0, 1, 0, 1, 0, 1], 1: [2, 3, 2, 3, 2, 3, 2, 3]}, index=expected_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_unordered_multiindex(self, future_stack):
        values = np.arange(5)
        data = np.vstack([[f'b{x}' for x in values], [f'a{x}' for x in values]])
        df = DataFrame(data.T, columns=['b', 'a'])
        df.columns.name = 'first'
        second_level_dict = {'x': df}
        multi_level_df = pd.concat(second_level_dict, axis=1)
        multi_level_df.columns.names = ['second', 'first']
        df = multi_level_df.reindex(sorted(multi_level_df.columns), axis=1)
        result = df.stack(['first', 'second'], future_stack=future_stack).unstack(['first', 'second'])
        expected = DataFrame([['a0', 'b0'], ['a1', 'b1'], ['a2', 'b2'], ['a3', 'b3'], ['a4', 'b4']], index=[0, 1, 2, 3, 4], columns=MultiIndex.from_tuples([('a', 'x'), ('b', 'x')], names=['first', 'second']))
        tm.assert_frame_equal(result, expected)

    def test_unstack_preserve_types(self, multiindex_year_month_day_dataframe_random_data, using_infer_string):
        ymd = multiindex_year_month_day_dataframe_random_data
        ymd['E'] = 'foo'
        ymd['F'] = 2
        unstacked = ymd.unstack('month')
        assert unstacked['A', 1].dtype == np.float64
        assert unstacked['E', 1].dtype == np.object_ if not using_infer_string else 'string'
        assert unstacked['F', 1].dtype == np.float64

    def test_unstack_group_index_overflow(self, future_stack):
        codes = np.tile(np.arange(500), 2)
        level = np.arange(500)
        index = MultiIndex(levels=[level] * 8 + [[0, 1]], codes=[codes] * 8 + [np.arange(2).repeat(500)])
        s = Series(np.arange(1000), index=index)
        result = s.unstack()
        assert result.shape == (500, 2)
        stacked = result.stack(future_stack=future_stack)
        tm.assert_series_equal(s, stacked.reindex(s.index))
        index = MultiIndex(levels=[[0, 1]] + [level] * 8, codes=[np.arange(2).repeat(500)] + [codes] * 8)
        s = Series(np.arange(1000), index=index)
        result = s.unstack(0)
        assert result.shape == (500, 2)
        index = MultiIndex(levels=[level] * 4 + [[0, 1]] + [level] * 4, codes=[codes] * 4 + [np.arange(2).repeat(500)] + [codes] * 4)
        s = Series(np.arange(1000), index=index)
        result = s.unstack(4)
        assert result.shape == (500, 2)

    def test_unstack_with_missing_int_cast_to_float(self, using_array_manager):
        df = DataFrame({'a': ['A', 'A', 'B'], 'b': ['ca', 'cb', 'cb'], 'v': [10] * 3}).set_index(['a', 'b'])
        df['is_'] = 1
        if not using_array_manager:
            assert len(df._mgr.blocks) == 2
        result = df.unstack('b')
        result['is_', 'ca'] = result['is_', 'ca'].fillna(0)
        expected = DataFrame([[10.0, 10.0, 1.0, 1.0], [np.nan, 10.0, 0.0, 1.0]], index=Index(['A', 'B'], name='a'), columns=MultiIndex.from_tuples([('v', 'ca'), ('v', 'cb'), ('is_', 'ca'), ('is_', 'cb')], names=[None, 'b']))
        if using_array_manager:
            expected['v', 'cb'] = expected['v', 'cb'].astype('int64')
            expected['is_', 'cb'] = expected['is_', 'cb'].astype('int64')
        tm.assert_frame_equal(result, expected)

    def test_unstack_with_level_has_nan(self):
        df1 = DataFrame({'L1': [1, 2, 3, 4], 'L2': [3, 4, 1, 2], 'L3': [1, 1, 1, 1], 'x': [1, 2, 3, 4]})
        df1 = df1.set_index(['L1', 'L2', 'L3'])
        new_levels = ['n1', 'n2', 'n3', None]
        df1.index = df1.index.set_levels(levels=new_levels, level='L1')
        df1.index = df1.index.set_levels(levels=new_levels, level='L2')
        result = df1.unstack('L3')['x', 1].sort_index().index
        expected = MultiIndex(levels=[['n1', 'n2', 'n3', None], ['n1', 'n2', 'n3', None]], codes=[[0, 1, 2, 3], [2, 3, 0, 1]], names=['L1', 'L2'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nan_in_multiindex_columns(self, future_stack):
        df = DataFrame(np.zeros([1, 5]), columns=MultiIndex.from_tuples([(0, None, None), (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1)]))
        result = df.stack(2, future_stack=future_stack)
        if future_stack:
            index = MultiIndex(levels=[[0], [0.0, 1.0]], codes=[[0, 0, 0], [-1, 0, 1]])
            columns = MultiIndex(levels=[[0], [2, 3]], codes=[[0, 0, 0], [-1, 0, 1]])
        else:
            index = Index([(0, None), (0, 0), (0, 1)])
            columns = Index([(0, None), (0, 2), (0, 3)])
        expected = DataFrame([[0.0, np.nan, np.nan], [np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0]], index=index, columns=columns)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_multi_level_stack_categorical(self, future_stack):
        midx = MultiIndex.from_arrays([['A'] * 2 + ['B'] * 2, pd.Categorical(list('abab')), pd.Categorical(list('ccdd'))])
        df = DataFrame(np.arange(8).reshape(2, 4), columns=midx)
        result = df.stack([1, 2], future_stack=future_stack)
        if future_stack:
            expected = DataFrame([[0, np.nan], [1, np.nan], [np.nan, 2], [np.nan, 3], [4, np.nan], [5, np.nan], [np.nan, 6], [np.nan, 7]], columns=['A', 'B'], index=MultiIndex.from_arrays([[0] * 4 + [1] * 4, pd.Categorical(list('abababab')), pd.Categorical(list('ccddccdd'))]))
        else:
            expected = DataFrame([[0, np.nan], [np.nan, 2], [1, np.nan], [np.nan, 3], [4, np.nan], [np.nan, 6], [5, np.nan], [np.nan, 7]], columns=['A', 'B'], index=MultiIndex.from_arrays([[0] * 4 + [1] * 4, pd.Categorical(list('aabbaabb')), pd.Categorical(list('cdcdcdcd'))]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nan_level(self, future_stack):
        df_nan = DataFrame(np.arange(4).reshape(2, 2), columns=MultiIndex.from_tuples([('A', np.nan), ('B', 'b')], names=['Upper', 'Lower']), index=Index([0, 1], name='Num'), dtype=np.float64)
        result = df_nan.stack(future_stack=future_stack)
        if future_stack:
            index = MultiIndex(levels=[[0, 1], [np.nan, 'b']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=['Num', 'Lower'])
        else:
            index = MultiIndex.from_tuples([(0, np.nan), (0, 'b'), (1, np.nan), (1, 'b')], names=['Num', 'Lower'])
        expected = DataFrame([[0.0, np.nan], [np.nan, 1], [2.0, np.nan], [np.nan, 3.0]], columns=Index(['A', 'B'], name='Upper'), index=index)
        tm.assert_frame_equal(result, expected)

    def test_unstack_categorical_columns(self):
        idx = MultiIndex.from_product([['A'], [0, 1]])
        df = DataFrame({'cat': pd.Categorical(['a', 'b'])}, index=idx)
        result = df.unstack()
        expected = DataFrame({0: pd.Categorical(['a'], categories=['a', 'b']), 1: pd.Categorical(['b'], categories=['a', 'b'])}, index=['A'])
        expected.columns = MultiIndex.from_tuples([('cat', 0), ('cat', 1)])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unsorted(self, future_stack):
        PAE = ['ITA', 'FRA']
        VAR = ['A1', 'A2']
        TYP = ['CRT', 'DBT', 'NET']
        MI = MultiIndex.from_product([PAE, VAR, TYP], names=['PAE', 'VAR', 'TYP'])
        V = list(range(len(MI)))
        DF = DataFrame(data=V, index=MI, columns=['VALUE'])
        DF = DF.unstack(['VAR', 'TYP'])
        DF.columns = DF.columns.droplevel(0)
        DF.loc[:, ('A0', 'NET')] = 9999
        result = DF.stack(['VAR', 'TYP'], future_stack=future_stack).sort_index()
        expected = DF.sort_index(axis=1).stack(['VAR', 'TYP'], future_stack=future_stack).sort_index()
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nullable_dtype(self, future_stack):
        columns = MultiIndex.from_product([['54511', '54515'], ['r', 't_mean']], names=['station', 'element'])
        index = Index([1, 2, 3], name='time')
        arr = np.array([[50, 226, 10, 215], [10, 215, 9, 220], [305, 232, 111, 220]])
        df = DataFrame(arr, columns=columns, index=index, dtype=pd.Int64Dtype())
        result = df.stack('station', future_stack=future_stack)
        expected = df.astype(np.int64).stack('station', future_stack=future_stack).astype(pd.Int64Dtype())
        tm.assert_frame_equal(result, expected)
        df[df.columns[0]] = df[df.columns[0]].astype(pd.Float64Dtype())
        result = df.stack('station', future_stack=future_stack)
        expected = DataFrame({'r': pd.array([50.0, 10.0, 10.0, 9.0, 305.0, 111.0], dtype=pd.Float64Dtype()), 't_mean': pd.array([226, 215, 215, 220, 232, 220], dtype=pd.Int64Dtype())}, index=MultiIndex.from_product([index, columns.levels[0]]))
        expected.columns.name = 'element'
        tm.assert_frame_equal(result, expected)

    def test_unstack_mixed_level_names(self):
        arrays = [['a', 'a'], [1, 2], ['red', 'blue']]
        idx = MultiIndex.from_arrays(arrays, names=('x', 0, 'y'))
        df = DataFrame({'m': [1, 2]}, index=idx)
        result = df.unstack('x')
        expected = DataFrame([[1], [2]], columns=MultiIndex.from_tuples([('m', 'a')], names=[None, 'x']), index=MultiIndex.from_tuples([(1, 'red'), (2, 'blue')], names=[0, 'y']))
        tm.assert_frame_equal(result, expected)