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
class TestDataFrameReshape:

    def test_stack_unstack(self, float_frame, future_stack):
        df = float_frame.copy()
        df[:] = np.arange(np.prod(df.shape)).reshape(df.shape)
        stacked = df.stack(future_stack=future_stack)
        stacked_df = DataFrame({'foo': stacked, 'bar': stacked})
        unstacked = stacked.unstack()
        unstacked_df = stacked_df.unstack()
        tm.assert_frame_equal(unstacked, df)
        tm.assert_frame_equal(unstacked_df['bar'], df)
        unstacked_cols = stacked.unstack(0)
        unstacked_cols_df = stacked_df.unstack(0)
        tm.assert_frame_equal(unstacked_cols.T, df)
        tm.assert_frame_equal(unstacked_cols_df['bar'].T, df)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_level(self, future_stack):
        levels = [range(3), [3, 'a', 'b'], [1, 2]]
        df = DataFrame(1, index=levels[0], columns=levels[1])
        result = df.stack(future_stack=future_stack)
        expected = Series(1, index=MultiIndex.from_product(levels[:2]))
        tm.assert_series_equal(result, expected)
        df = DataFrame(1, index=levels[0], columns=MultiIndex.from_product(levels[1:]))
        result = df.stack(1, future_stack=future_stack)
        expected = DataFrame(1, index=MultiIndex.from_product([levels[0], levels[2]]), columns=levels[1])
        tm.assert_frame_equal(result, expected)
        result = df[['a', 'b']].stack(1, future_stack=future_stack)
        expected = expected[['a', 'b']]
        tm.assert_frame_equal(result, expected)

    def test_unstack_not_consolidated(self, using_array_manager):
        df = DataFrame({'x': [1, 2, np.nan], 'y': [3.0, 4, np.nan]})
        df2 = df[['x']]
        df2['y'] = df['y']
        if not using_array_manager:
            assert len(df2._mgr.blocks) == 2
        res = df2.unstack()
        expected = df.unstack()
        tm.assert_series_equal(res, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_fill(self, future_stack):
        data = Series([1, 2, 4, 5], dtype=np.int16)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack(fill_value=-1)
        expected = DataFrame({'a': [1, -1, 5], 'b': [2, 4, -1]}, index=['x', 'y', 'z'], dtype=np.int16)
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=0.5)
        expected = DataFrame({'a': [1, 0.5, 5], 'b': [2, 4, 0.5]}, index=['x', 'y', 'z'], dtype=float)
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'x': ['a', 'a', 'b'], 'y': ['j', 'k', 'j'], 'z': [0, 1, 2], 'w': [0, 1, 2]}).set_index(['x', 'y', 'z'])
        unstacked = df.unstack(['x', 'y'], fill_value=0)
        key = ('w', 'b', 'j')
        expected = unstacked[key]
        result = Series([0, 0, 2], index=unstacked.index, name=key)
        tm.assert_series_equal(result, expected)
        stacked = unstacked.stack(['x', 'y'], future_stack=future_stack)
        stacked.index = stacked.index.reorder_levels(df.index.names)
        stacked = stacked.astype(np.int64)
        result = stacked.loc[df.index]
        tm.assert_frame_equal(result, df)
        s = df['w']
        result = s.unstack(['x', 'y'], fill_value=0)
        expected = unstacked['w']
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame(self):
        rows = [[1, 2], [3, 4], [5, 6], [7, 8]]
        df = DataFrame(rows, columns=list('AB'), dtype=np.int32)
        df.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = df.unstack(fill_value=-1)
        rows = [[1, 3, 2, 4], [-1, 5, -1, 6], [7, -1, 8, -1]]
        expected = DataFrame(rows, index=list('xyz'), dtype=np.int32)
        expected.columns = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
        tm.assert_frame_equal(result, expected)
        df['A'] = df['A'].astype(np.int16)
        df['B'] = df['B'].astype(np.float64)
        result = df.unstack(fill_value=-1)
        expected['A'] = expected['A'].astype(np.int16)
        expected['B'] = expected['B'].astype(np.float64)
        tm.assert_frame_equal(result, expected)
        result = df.unstack(fill_value=0.5)
        rows = [[1, 3, 2, 4], [0.5, 5, 0.5, 6], [7, 0.5, 8, 0.5]]
        expected = DataFrame(rows, index=list('xyz'), dtype=float)
        expected.columns = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_datetime(self):
        dv = date_range('2012-01-01', periods=4).values
        data = Series(dv)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': [dv[0], pd.NaT, dv[3]], 'b': [dv[1], dv[2], pd.NaT]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=dv[0])
        expected = DataFrame({'a': [dv[0], dv[0], dv[3]], 'b': [dv[1], dv[2], dv[0]]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_timedelta(self):
        td = [Timedelta(days=i) for i in range(4)]
        data = Series(td)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': [td[0], pd.NaT, td[3]], 'b': [td[1], td[2], pd.NaT]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=td[1])
        expected = DataFrame({'a': [td[0], td[1], td[3]], 'b': [td[1], td[2], td[1]]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_period(self):
        periods = [Period('2012-01'), Period('2012-02'), Period('2012-03'), Period('2012-04')]
        data = Series(periods)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': [periods[0], None, periods[3]], 'b': [periods[1], periods[2], None]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=periods[1])
        expected = DataFrame({'a': [periods[0], periods[1], periods[3]], 'b': [periods[1], periods[2], periods[1]]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_categorical(self):
        data = Series(['a', 'b', 'c', 'a'], dtype='category')
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': pd.Categorical(list('axa'), categories=list('abc')), 'b': pd.Categorical(list('bcx'), categories=list('abc'))}, index=list('xyz'))
        tm.assert_frame_equal(result, expected)
        msg = 'Cannot setitem on a Categorical with a new category \\(d\\)'
        with pytest.raises(TypeError, match=msg):
            data.unstack(fill_value='d')
        result = data.unstack(fill_value='c')
        expected = DataFrame({'a': pd.Categorical(list('aca'), categories=list('abc')), 'b': pd.Categorical(list('bcc'), categories=list('abc'))}, index=list('xyz'))
        tm.assert_frame_equal(result, expected)

    def test_unstack_tuplename_in_multiindex(self):
        idx = MultiIndex.from_product([['a', 'b', 'c'], [1, 2, 3]], names=[('A', 'a'), ('B', 'b')])
        df = DataFrame({'d': [1] * 9, 'e': [2] * 9}, index=idx)
        result = df.unstack(('A', 'a'))
        expected = DataFrame([[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]], columns=MultiIndex.from_tuples([('d', 'a'), ('d', 'b'), ('d', 'c'), ('e', 'a'), ('e', 'b'), ('e', 'c')], names=[None, ('A', 'a')]), index=Index([1, 2, 3], name=('B', 'b')))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('unstack_idx, expected_values, expected_index, expected_columns', [(('A', 'a'), [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=['B', 'C']), MultiIndex.from_tuples([('d', 'a'), ('d', 'b'), ('e', 'a'), ('e', 'b')], names=[None, ('A', 'a')])), ((('A', 'a'), 'B'), [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2, 2]], Index([3, 4], name='C'), MultiIndex.from_tuples([('d', 'a', 1), ('d', 'a', 2), ('d', 'b', 1), ('d', 'b', 2), ('e', 'a', 1), ('e', 'a', 2), ('e', 'b', 1), ('e', 'b', 2)], names=[None, ('A', 'a'), 'B']))])
    def test_unstack_mixed_type_name_in_multiindex(self, unstack_idx, expected_values, expected_index, expected_columns):
        idx = MultiIndex.from_product([['a', 'b'], [1, 2], [3, 4]], names=[('A', 'a'), 'B', 'C'])
        df = DataFrame({'d': [1] * 8, 'e': [2] * 8}, index=idx)
        result = df.unstack(unstack_idx)
        expected = DataFrame(expected_values, columns=expected_columns, index=expected_index)
        tm.assert_frame_equal(result, expected)

    def test_unstack_preserve_dtypes(self):
        df = DataFrame({'state': ['IL', 'MI', 'NC'], 'index': ['a', 'b', 'c'], 'some_categories': Series(['a', 'b', 'c']).astype('category'), 'A': np.random.default_rng(2).random(3), 'B': 1, 'C': 'foo', 'D': pd.Timestamp('20010102'), 'E': Series([1.0, 50.0, 100.0]).astype('float32'), 'F': Series([3.0, 4.0, 5.0]).astype('float64'), 'G': False, 'H': Series([1, 200, 923442]).astype('int8')})

        def unstack_and_compare(df, column_name):
            unstacked1 = df.unstack([column_name])
            unstacked2 = df.unstack(column_name)
            tm.assert_frame_equal(unstacked1, unstacked2)
        df1 = df.set_index(['state', 'index'])
        unstack_and_compare(df1, 'index')
        df1 = df.set_index(['state', 'some_categories'])
        unstack_and_compare(df1, 'some_categories')
        df1 = df.set_index(['F', 'C'])
        unstack_and_compare(df1, 'F')
        df1 = df.set_index(['G', 'B', 'state'])
        unstack_and_compare(df1, 'B')
        df1 = df.set_index(['E', 'A'])
        unstack_and_compare(df1, 'E')
        df1 = df.set_index(['state', 'index'])
        s = df1['A']
        unstack_and_compare(s, 'index')

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_ints(self, future_stack):
        columns = MultiIndex.from_tuples(list(itertools.product(range(3), repeat=3)))
        df = DataFrame(np.random.default_rng(2).standard_normal((30, 27)), columns=columns)
        tm.assert_frame_equal(df.stack(level=[1, 2], future_stack=future_stack), df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))
        tm.assert_frame_equal(df.stack(level=[-2, -1], future_stack=future_stack), df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))
        df_named = df.copy()
        return_value = df_named.columns.set_names(range(3), inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df_named.stack(level=[1, 2], future_stack=future_stack), df_named.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_levels(self, future_stack):
        columns = MultiIndex.from_tuples([('A', 'cat', 'long'), ('B', 'cat', 'long'), ('A', 'dog', 'short'), ('B', 'dog', 'short')], names=['exp', 'animal', 'hair_length'])
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=columns)
        animal_hair_stacked = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
        exp_hair_stacked = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
        df2 = df.copy()
        df2.columns.names = ['exp', 'animal', 1]
        tm.assert_frame_equal(df2.stack(level=['animal', 1], future_stack=future_stack), animal_hair_stacked, check_names=False)
        tm.assert_frame_equal(df2.stack(level=['exp', 1], future_stack=future_stack), exp_hair_stacked, check_names=False)
        msg = 'level should contain all level names or all level numbers, not a mixture of the two'
        with pytest.raises(ValueError, match=msg):
            df2.stack(level=['animal', 0], future_stack=future_stack)
        df3 = df.copy()
        df3.columns.names = ['exp', 'animal', 0]
        tm.assert_frame_equal(df3.stack(level=['animal', 0], future_stack=future_stack), animal_hair_stacked, check_names=False)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_int_level_names(self, future_stack):
        columns = MultiIndex.from_tuples([('A', 'cat', 'long'), ('B', 'cat', 'long'), ('A', 'dog', 'short'), ('B', 'dog', 'short')], names=['exp', 'animal', 'hair_length'])
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=columns)
        exp_animal_stacked = df.stack(level=['exp', 'animal'], future_stack=future_stack)
        animal_hair_stacked = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
        exp_hair_stacked = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
        df2 = df.copy()
        df2.columns.names = [0, 1, 2]
        tm.assert_frame_equal(df2.stack(level=[1, 2], future_stack=future_stack), animal_hair_stacked, check_names=False)
        tm.assert_frame_equal(df2.stack(level=[0, 1], future_stack=future_stack), exp_animal_stacked, check_names=False)
        tm.assert_frame_equal(df2.stack(level=[0, 2], future_stack=future_stack), exp_hair_stacked, check_names=False)
        df3 = df.copy()
        df3.columns.names = [2, 0, 1]
        tm.assert_frame_equal(df3.stack(level=[0, 1], future_stack=future_stack), animal_hair_stacked, check_names=False)
        tm.assert_frame_equal(df3.stack(level=[2, 0], future_stack=future_stack), exp_animal_stacked, check_names=False)
        tm.assert_frame_equal(df3.stack(level=[2, 1], future_stack=future_stack), exp_hair_stacked, check_names=False)

    def test_unstack_bool(self):
        df = DataFrame([False, False], index=MultiIndex.from_arrays([['a', 'b'], ['c', 'l']]), columns=['col'])
        rs = df.unstack()
        xp = DataFrame(np.array([[False, np.nan], [np.nan, False]], dtype=object), index=['a', 'b'], columns=MultiIndex.from_arrays([['col', 'col'], ['c', 'l']]))
        tm.assert_frame_equal(rs, xp)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_level_binding(self, future_stack):
        mi = MultiIndex(levels=[['foo', 'bar'], ['one', 'two'], ['a', 'b']], codes=[[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]], names=['first', 'second', 'third'])
        s = Series(0, index=mi)
        result = s.unstack([1, 2]).stack(0, future_stack=future_stack)
        expected_mi = MultiIndex(levels=[['foo', 'bar'], ['one', 'two']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=['first', 'second'])
        expected = DataFrame(np.array([[0, np.nan], [np.nan, 0], [0, np.nan], [np.nan, 0]], dtype=np.float64), index=expected_mi, columns=Index(['b', 'a'], name='third'))
        tm.assert_frame_equal(result, expected)

    def test_unstack_to_series(self, float_frame):
        data = float_frame.unstack()
        assert isinstance(data, Series)
        undo = data.unstack().T
        tm.assert_frame_equal(undo, float_frame)
        data = DataFrame({'x': [1, 2, np.nan], 'y': [3.0, 4, np.nan]})
        data.index = Index(['a', 'b', 'c'])
        result = data.unstack()
        midx = MultiIndex(levels=[['x', 'y'], ['a', 'b', 'c']], codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        expected = Series([1, 2, np.nan, 3, 4, np.nan], index=midx)
        tm.assert_series_equal(result, expected)
        old_data = data.copy()
        for _ in range(4):
            data = data.unstack()
        tm.assert_frame_equal(old_data, data)

    def test_unstack_dtypes(self, using_infer_string):
        rows = [[1, 1, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4], [2, 2, 3, 4]]
        df = DataFrame(rows, columns=list('ABCD'))
        result = df.dtypes
        expected = Series([np.dtype('int64')] * 4, index=list('ABCD'))
        tm.assert_series_equal(result, expected)
        df2 = df.set_index(['A', 'B'])
        df3 = df2.unstack('B')
        result = df3.dtypes
        expected = Series([np.dtype('int64')] * 4, index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B')))
        tm.assert_series_equal(result, expected)
        df2 = df.set_index(['A', 'B'])
        df2['C'] = 3.0
        df3 = df2.unstack('B')
        result = df3.dtypes
        expected = Series([np.dtype('float64')] * 2 + [np.dtype('int64')] * 2, index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B')))
        tm.assert_series_equal(result, expected)
        df2['D'] = 'foo'
        df3 = df2.unstack('B')
        result = df3.dtypes
        dtype = 'string' if using_infer_string else np.dtype('object')
        expected = Series([np.dtype('float64')] * 2 + [dtype] * 2, index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B')))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('c, d', ((np.zeros(5), np.zeros(5)), (np.arange(5, dtype='f8'), np.arange(5, 10, dtype='f8'))))
    def test_unstack_dtypes_mixed_date(self, c, d):
        df = DataFrame({'A': ['a'] * 5, 'C': c, 'D': d, 'B': date_range('2012-01-01', periods=5)})
        right = df.iloc[:3].copy(deep=True)
        df = df.set_index(['A', 'B'])
        df['D'] = df['D'].astype('int64')
        left = df.iloc[:3].unstack(0)
        right = right.set_index(['A', 'B']).unstack(0)
        right['D', 'a'] = right['D', 'a'].astype('int64')
        assert left.shape == (3, 2)
        tm.assert_frame_equal(left, right)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_non_unique_index_names(self, future_stack):
        idx = MultiIndex.from_tuples([('a', 'b'), ('c', 'd')], names=['c1', 'c1'])
        df = DataFrame([1, 2], index=idx)
        msg = 'The name c1 occurs multiple times, use a level number'
        with pytest.raises(ValueError, match=msg):
            df.unstack('c1')
        with pytest.raises(ValueError, match=msg):
            df.T.stack('c1', future_stack=future_stack)

    def test_unstack_unused_levels(self):
        idx = MultiIndex.from_product([['a'], ['A', 'B', 'C', 'D']])[:-1]
        df = DataFrame([[1, 0]] * 3, index=idx)
        result = df.unstack()
        exp_col = MultiIndex.from_product([[0, 1], ['A', 'B', 'C']])
        expected = DataFrame([[1, 1, 1, 0, 0, 0]], index=['a'], columns=exp_col)
        tm.assert_frame_equal(result, expected)
        assert (result.columns.levels[1] == idx.levels[1]).all()
        levels = [[0, 1, 7], [0, 1, 2, 3]]
        codes = [[0, 0, 1, 1], [0, 2, 0, 2]]
        idx = MultiIndex(levels, codes)
        block = np.arange(4).reshape(2, 2)
        df = DataFrame(np.concatenate([block, block + 4]), index=idx)
        result = df.unstack()
        expected = DataFrame(np.concatenate([block * 2, block * 2 + 1], axis=1), columns=idx)
        tm.assert_frame_equal(result, expected)
        assert (result.columns.levels[1] == idx.levels[1]).all()

    @pytest.mark.parametrize('level, idces, col_level, idx_level', ((0, [13, 16, 6, 9, 2, 5, 8, 11], [np.nan, 'a', 2], [np.nan, 5, 1]), (1, [8, 11, 1, 4, 12, 15, 13, 16], [np.nan, 5, 1], [np.nan, 'a', 2])))
    def test_unstack_unused_levels_mixed_with_nan(self, level, idces, col_level, idx_level):
        levels = [['a', 2, 'c'], [1, 3, 5, 7]]
        codes = [[0, -1, 1, 1], [0, 2, -1, 2]]
        idx = MultiIndex(levels, codes)
        data = np.arange(8)
        df = DataFrame(data.reshape(4, 2), index=idx)
        result = df.unstack(level=level)
        exp_data = np.zeros(18) * np.nan
        exp_data[idces] = data
        cols = MultiIndex.from_product([[0, 1], col_level])
        expected = DataFrame(exp_data.reshape(3, 6), index=idx_level, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('cols', [['A', 'C'], slice(None)])
    def test_unstack_unused_level(self, cols):
        df = DataFrame([[2010, 'a', 'I'], [2011, 'b', 'II']], columns=['A', 'B', 'C'])
        ind = df.set_index(['A', 'B', 'C'], drop=False)
        selection = ind.loc[(slice(None), slice(None), 'I'), cols]
        result = selection.unstack()
        expected = ind.iloc[[0]][cols]
        expected.columns = MultiIndex.from_product([expected.columns, ['I']], names=[None, 'C'])
        expected.index = expected.index.droplevel('C')
        tm.assert_frame_equal(result, expected)

    def test_unstack_long_index(self):
        df = DataFrame([[1]], columns=MultiIndex.from_tuples([[0]], names=['c1']), index=MultiIndex.from_tuples([[0, 0, 1, 0, 0, 0, 1]], names=['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']))
        result = df.unstack(['i2', 'i3', 'i4', 'i5', 'i6', 'i7'])
        expected = DataFrame([[1]], columns=MultiIndex.from_tuples([[0, 0, 1, 0, 0, 0, 1]], names=['c1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']), index=Index([0], name='i1'))
        tm.assert_frame_equal(result, expected)

    def test_unstack_multi_level_cols(self):
        df = DataFrame([[0.0, 0.0], [0.0, 0.0]], columns=MultiIndex.from_tuples([['B', 'C'], ['B', 'D']], names=['c1', 'c2']), index=MultiIndex.from_tuples([[10, 20, 30], [10, 20, 40]], names=['i1', 'i2', 'i3']))
        assert df.unstack(['i2', 'i1']).columns.names[-2:] == ['i2', 'i1']

    def test_unstack_multi_level_rows_and_cols(self):
        df = DataFrame([[1, 2], [3, 4], [-1, -2], [-3, -4]], columns=MultiIndex.from_tuples([['a', 'b', 'c'], ['d', 'e', 'f']]), index=MultiIndex.from_tuples([['m1', 'P3', 222], ['m1', 'A5', 111], ['m2', 'P3', 222], ['m2', 'A5', 111]], names=['i1', 'i2', 'i3']))
        result = df.unstack(['i3', 'i2'])
        expected = df.unstack(['i3']).unstack(['i2'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('idx', [('jim', 'joe'), ('joe', 'jim')])
    @pytest.mark.parametrize('lev', list(range(2)))
    def test_unstack_nan_index1(self, idx, lev):

        def cast(val):
            val_str = '' if val != val else val
            return f'{val_str:1}'
        df = DataFrame({'jim': ['a', 'b', np.nan, 'd'], 'joe': ['w', 'x', 'y', 'z'], 'jolie': ['a.w', 'b.x', ' .y', 'd.z']})
        left = df.set_index(['jim', 'joe']).unstack()['jolie']
        right = df.set_index(['joe', 'jim']).unstack()['jolie'].T
        tm.assert_frame_equal(left, right)
        mi = df.set_index(list(idx))
        udf = mi.unstack(level=lev)
        assert udf.notna().values.sum() == len(df)
        mk_list = lambda a: list(a) if isinstance(a, tuple) else [a]
        rows, cols = udf['jolie'].notna().values.nonzero()
        for i, j in zip(rows, cols):
            left = sorted(udf['jolie'].iloc[i, j].split('.'))
            right = mk_list(udf['jolie'].index[i]) + mk_list(udf['jolie'].columns[j])
            right = sorted(map(cast, right))
            assert left == right

    @pytest.mark.parametrize('idx', itertools.permutations(['1st', '2nd', '3rd']))
    @pytest.mark.parametrize('lev', list(range(3)))
    @pytest.mark.parametrize('col', ['4th', '5th'])
    def test_unstack_nan_index_repeats(self, idx, lev, col):

        def cast(val):
            val_str = '' if val != val else val
            return f'{val_str:1}'
        df = DataFrame({'1st': ['d'] * 3 + [np.nan] * 5 + ['a'] * 2 + ['c'] * 3 + ['e'] * 2 + ['b'] * 5, '2nd': ['y'] * 2 + ['w'] * 3 + [np.nan] * 3 + ['z'] * 4 + [np.nan] * 3 + ['x'] * 3 + [np.nan] * 2, '3rd': [67, 39, 53, 72, 57, 80, 31, 18, 11, 30, 59, 50, 62, 59, 76, 52, 14, 53, 60, 51]})
        df['4th'], df['5th'] = (df.apply(lambda r: '.'.join(map(cast, r)), axis=1), df.apply(lambda r: '.'.join(map(cast, r.iloc[::-1])), axis=1))
        mi = df.set_index(list(idx))
        udf = mi.unstack(level=lev)
        assert udf.notna().values.sum() == 2 * len(df)
        mk_list = lambda a: list(a) if isinstance(a, tuple) else [a]
        rows, cols = udf[col].notna().values.nonzero()
        for i, j in zip(rows, cols):
            left = sorted(udf[col].iloc[i, j].split('.'))
            right = mk_list(udf[col].index[i]) + mk_list(udf[col].columns[j])
            right = sorted(map(cast, right))
            assert left == right

    def test_unstack_nan_index2(self):
        df = DataFrame({'A': list('aaaabbbb'), 'B': range(8), 'C': range(8)})
        df = df.astype({'B': 'float'})
        df.iloc[3, 1] = np.nan
        left = df.set_index(['A', 'B']).unstack(0)
        vals = [[3, 0, 1, 2, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, 4, 5, 6, 7]]
        vals = list(map(list, zip(*vals)))
        idx = Index([np.nan, 0, 1, 2, 4, 5, 6, 7], name='B')
        cols = MultiIndex(levels=[['C'], ['a', 'b']], codes=[[0, 0], [0, 1]], names=[None, 'A'])
        right = DataFrame(vals, columns=cols, index=idx)
        tm.assert_frame_equal(left, right)
        df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2, 'C': range(8)})
        df = df.astype({'B': 'float'})
        df.iloc[2, 1] = np.nan
        left = df.set_index(['A', 'B']).unstack(0)
        vals = [[2, np.nan], [0, 4], [1, 5], [np.nan, 6], [3, 7]]
        cols = MultiIndex(levels=[['C'], ['a', 'b']], codes=[[0, 0], [0, 1]], names=[None, 'A'])
        idx = Index([np.nan, 0, 1, 2, 3], name='B')
        right = DataFrame(vals, columns=cols, index=idx)
        tm.assert_frame_equal(left, right)
        df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2, 'C': range(8)})
        df = df.astype({'B': 'float'})
        df.iloc[3, 1] = np.nan
        left = df.set_index(['A', 'B']).unstack(0)
        vals = [[3, np.nan], [0, 4], [1, 5], [2, 6], [np.nan, 7]]
        cols = MultiIndex(levels=[['C'], ['a', 'b']], codes=[[0, 0], [0, 1]], names=[None, 'A'])
        idx = Index([np.nan, 0, 1, 2, 3], name='B')
        right = DataFrame(vals, columns=cols, index=idx)
        tm.assert_frame_equal(left, right)

    def test_unstack_nan_index3(self, using_array_manager):
        df = DataFrame({'A': list('aaaaabbbbb'), 'B': date_range('2012-01-01', periods=5).tolist() * 2, 'C': np.arange(10)})
        df.iloc[3, 1] = np.nan
        left = df.set_index(['A', 'B']).unstack()
        vals = np.array([[3, 0, 1, 2, np.nan, 4], [np.nan, 5, 6, 7, 8, 9]])
        idx = Index(['a', 'b'], name='A')
        cols = MultiIndex(levels=[['C'], date_range('2012-01-01', periods=5)], codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]], names=[None, 'B'])
        right = DataFrame(vals, columns=cols, index=idx)
        if using_array_manager:
            cols = right.columns[[1, 2, 3, 5]]
            right[cols] = right[cols].astype(df['C'].dtype)
        tm.assert_frame_equal(left, right)

    def test_unstack_nan_index4(self):
        vals = [['Hg', np.nan, np.nan, 680585148], ['U', 0.0, np.nan, 680585148], ['Pb', 7.07e-06, np.nan, 680585148], ['Sn', 2.3614e-05, 0.0133, 680607017], ['Ag', 0.0, 0.0133, 680607017], ['Hg', -0.00015, 0.0133, 680607017]]
        df = DataFrame(vals, columns=['agent', 'change', 'dosage', 's_id'], index=[17263, 17264, 17265, 17266, 17267, 17268])
        left = df.copy().set_index(['s_id', 'dosage', 'agent']).unstack()
        vals = [[np.nan, np.nan, 7.07e-06, np.nan, 0.0], [0.0, -0.00015, np.nan, 2.3614e-05, np.nan]]
        idx = MultiIndex(levels=[[680585148, 680607017], [0.0133]], codes=[[0, 1], [-1, 0]], names=['s_id', 'dosage'])
        cols = MultiIndex(levels=[['change'], ['Ag', 'Hg', 'Pb', 'Sn', 'U']], codes=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]], names=[None, 'agent'])
        right = DataFrame(vals, columns=cols, index=idx)
        tm.assert_frame_equal(left, right)
        left = df.loc[17264:].copy().set_index(['s_id', 'dosage', 'agent'])
        tm.assert_frame_equal(left.unstack(), right)

    def test_unstack_nan_index5(self):
        df = DataFrame({'1st': [1, 2, 1, 2, 1, 2], '2nd': date_range('2014-02-01', periods=6, freq='D'), 'jim': 100 + np.arange(6), 'joe': (np.random.default_rng(2).standard_normal(6) * 10).round(2)})
        df['3rd'] = df['2nd'] - pd.Timestamp('2014-02-02')
        df.loc[1, '2nd'] = df.loc[3, '2nd'] = np.nan
        df.loc[1, '3rd'] = df.loc[4, '3rd'] = np.nan
        left = df.set_index(['1st', '2nd', '3rd']).unstack(['2nd', '3rd'])
        assert left.notna().values.sum() == 2 * len(df)
        for col in ['jim', 'joe']:
            for _, r in df.iterrows():
                key = (r['1st'], (col, r['2nd'], r['3rd']))
                assert r[col] == left.loc[key]

    def test_stack_datetime_column_multiIndex(self, future_stack):
        t = datetime(2014, 1, 1)
        df = DataFrame([1, 2, 3, 4], columns=MultiIndex.from_tuples([(t, 'A', 'B')]))
        warn = None if future_stack else FutureWarning
        msg = 'The previous implementation of stack is deprecated'
        with tm.assert_produces_warning(warn, match=msg):
            result = df.stack(future_stack=future_stack)
        eidx = MultiIndex.from_product([(0, 1, 2, 3), ('B',)])
        ecols = MultiIndex.from_tuples([(t, 'A')])
        expected = DataFrame([1, 2, 3, 4], index=eidx, columns=ecols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('multiindex_columns', [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 1], [0, 2], [0, 3], [0], [2], [4], [4, 3, 2, 1, 0], [3, 2, 1, 0], [4, 2, 1, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2], [1, 0], [2, 0], [3, 0]])
    @pytest.mark.parametrize('level', (-1, 0, 1, [0, 1], [1, 0]))
    def test_stack_partial_multiIndex(self, multiindex_columns, level, future_stack):
        dropna = False if not future_stack else lib.no_default
        full_multiindex = MultiIndex.from_tuples([('B', 'x'), ('B', 'z'), ('A', 'y'), ('C', 'x'), ('C', 'u')], names=['Upper', 'Lower'])
        multiindex = full_multiindex[multiindex_columns]
        df = DataFrame(np.arange(3 * len(multiindex)).reshape(3, len(multiindex)), columns=multiindex)
        result = df.stack(level=level, dropna=dropna, future_stack=future_stack)
        if isinstance(level, int) and (not future_stack):
            expected = df.stack(level=level, dropna=True, future_stack=future_stack)
            if isinstance(expected, Series):
                tm.assert_series_equal(result, expected)
            else:
                tm.assert_frame_equal(result, expected)
        df.columns = MultiIndex.from_tuples(df.columns.to_numpy(), names=df.columns.names)
        expected = df.stack(level=level, dropna=dropna, future_stack=future_stack)
        if isinstance(expected, Series):
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_full_multiIndex(self, future_stack):
        full_multiindex = MultiIndex.from_tuples([('B', 'x'), ('B', 'z'), ('A', 'y'), ('C', 'x'), ('C', 'u')], names=['Upper', 'Lower'])
        df = DataFrame(np.arange(6).reshape(2, 3), columns=full_multiindex[[0, 1, 3]])
        dropna = False if not future_stack else lib.no_default
        result = df.stack(dropna=dropna, future_stack=future_stack)
        expected = DataFrame([[0, 2], [1, np.nan], [3, 5], [4, np.nan]], index=MultiIndex(levels=[[0, 1], ['u', 'x', 'y', 'z']], codes=[[0, 0, 1, 1], [1, 3, 1, 3]], names=[None, 'Lower']), columns=Index(['B', 'C'], name='Upper'))
        expected['B'] = expected['B'].astype(df.dtypes.iloc[0])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [False, True])
    def test_stack_preserve_categorical_dtype(self, ordered, future_stack):
        cidx = pd.CategoricalIndex(list('yxz'), categories=list('xyz'), ordered=ordered)
        df = DataFrame([[10, 11, 12]], columns=cidx)
        result = df.stack(future_stack=future_stack)
        midx = MultiIndex.from_product([df.index, cidx])
        expected = Series([10, 11, 12], index=midx)
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('ordered', [False, True])
    @pytest.mark.parametrize('labels,data', [(list('xyz'), [10, 11, 12, 13, 14, 15]), (list('zyx'), [14, 15, 12, 13, 10, 11])])
    def test_stack_multi_preserve_categorical_dtype(self, ordered, labels, data, future_stack):
        cidx = pd.CategoricalIndex(labels, categories=sorted(labels), ordered=ordered)
        cidx2 = pd.CategoricalIndex(['u', 'v'], ordered=ordered)
        midx = MultiIndex.from_product([cidx, cidx2])
        df = DataFrame([sorted(data)], columns=midx)
        result = df.stack([0, 1], future_stack=future_stack)
        labels = labels if future_stack else sorted(labels)
        s_cidx = pd.CategoricalIndex(labels, ordered=ordered)
        expected_data = sorted(data) if future_stack else data
        expected = Series(expected_data, index=MultiIndex.from_product([[0], s_cidx, cidx2]))
        tm.assert_series_equal(result, expected)

    def test_stack_preserve_categorical_dtype_values(self, future_stack):
        cat = pd.Categorical(['a', 'a', 'b', 'c'])
        df = DataFrame({'A': cat, 'B': cat})
        result = df.stack(future_stack=future_stack)
        index = MultiIndex.from_product([[0, 1, 2, 3], ['A', 'B']])
        expected = Series(pd.Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c']), index=index)
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
    @pytest.mark.parametrize('index, columns', [([0, 0, 1, 1], MultiIndex.from_product([[1, 2], ['a', 'b']])), ([0, 0, 2, 3], MultiIndex.from_product([[1, 2], ['a', 'b']])), ([0, 1, 2, 3], MultiIndex.from_product([[1, 2], ['a', 'b']]))])
    def test_stack_multi_columns_non_unique_index(self, index, columns, future_stack):
        df = DataFrame(index=index, columns=columns).fillna(1)
        stacked = df.stack(future_stack=future_stack)
        new_index = MultiIndex.from_tuples(stacked.index.to_numpy())
        expected = DataFrame(stacked.to_numpy(), index=new_index, columns=stacked.columns)
        tm.assert_frame_equal(stacked, expected)
        stacked_codes = np.asarray(stacked.index.codes)
        expected_codes = np.asarray(new_index.codes)
        tm.assert_numpy_array_equal(stacked_codes, expected_codes)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('vals1, vals2, dtype1, dtype2, expected_dtype', [([1, 2], [3.0, 4.0], 'Int64', 'Float64', 'Float64'), ([1, 2], ['foo', 'bar'], 'Int64', 'string', 'object')])
    def test_stack_multi_columns_mixed_extension_types(self, vals1, vals2, dtype1, dtype2, expected_dtype, future_stack):
        df = DataFrame({('A', 1): Series(vals1, dtype=dtype1), ('A', 2): Series(vals2, dtype=dtype2)})
        result = df.stack(future_stack=future_stack)
        expected = df.astype(object).stack(future_stack=future_stack).astype(expected_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('level', [0, 1])
    def test_unstack_mixed_extension_types(self, level):
        index = MultiIndex.from_tuples([('A', 0), ('A', 1), ('B', 1)], names=['a', 'b'])
        df = DataFrame({'A': pd.array([0, 1, None], dtype='Int64'), 'B': pd.Categorical(['a', 'a', 'b'])}, index=index)
        result = df.unstack(level=level)
        expected = df.astype(object).unstack(level=level)
        if level == 0:
            expected['A', 'B'] = expected['A', 'B'].fillna(pd.NA)
        else:
            expected['A', 0] = expected['A', 0].fillna(pd.NA)
        expected_dtypes = Series([df.A.dtype] * 2 + [df.B.dtype] * 2, index=result.columns)
        tm.assert_series_equal(result.dtypes, expected_dtypes)
        tm.assert_frame_equal(result.astype(object), expected)

    @pytest.mark.parametrize('level', [0, 'baz'])
    def test_unstack_swaplevel_sortlevel(self, level):
        mi = MultiIndex.from_product([[0], ['d', 'c']], names=['bar', 'baz'])
        df = DataFrame([[0, 2], [1, 3]], index=mi, columns=['B', 'A'])
        df.columns.name = 'foo'
        expected = DataFrame([[3, 1, 2, 0]], columns=MultiIndex.from_tuples([('c', 'A'), ('c', 'B'), ('d', 'A'), ('d', 'B')], names=['baz', 'foo']))
        expected.index.name = 'bar'
        result = df.unstack().swaplevel(axis=1).sort_index(axis=1, level=level)
        tm.assert_frame_equal(result, expected)