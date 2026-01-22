from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
class TestGrouping:

    @pytest.mark.parametrize('index', [Index(list('abcde')), Index(np.arange(5)), Index(np.arange(5, dtype=float)), date_range('2020-01-01', periods=5), period_range('2020-01-01', periods=5)])
    def test_grouper_index_types(self, index):
        df = DataFrame(np.arange(10).reshape(5, 2), columns=list('AB'), index=index)
        df.groupby(list('abcde'), group_keys=False).apply(lambda x: x)
        df.index = df.index[::-1]
        df.groupby(list('abcde'), group_keys=False).apply(lambda x: x)

    def test_grouper_multilevel_freq(self):
        d0 = date.today() - timedelta(days=14)
        dates = date_range(d0, date.today())
        date_index = MultiIndex.from_product([dates, dates], names=['foo', 'bar'])
        df = DataFrame(np.random.default_rng(2).integers(0, 100, 225), index=date_index)
        expected = df.reset_index().groupby([Grouper(key='foo', freq='W'), Grouper(key='bar', freq='W')]).sum()
        expected.columns = Index([0], dtype='int64')
        result = df.groupby([Grouper(level='foo', freq='W'), Grouper(level='bar', freq='W')]).sum()
        tm.assert_frame_equal(result, expected)
        result = df.groupby([Grouper(level=0, freq='W'), Grouper(level=1, freq='W')]).sum()
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug(self):
        df = DataFrame({'A': [0, 0, 1, 1, 2, 2], 'B': [1, 2, 3, 4, 5, 6]})
        g = df.groupby('A')
        expected = g.sum()
        g = df.groupby(Grouper(key='A'))
        result = g.sum()
        tm.assert_frame_equal(result, expected)
        msg = 'Grouper axis keyword is deprecated and will be removed'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gpr = Grouper(key='A', axis=0)
        g = df.groupby(gpr)
        result = g.sum()
        tm.assert_frame_equal(result, expected)
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            result = g.apply(lambda x: x.sum())
        expected['A'] = [0, 2, 4]
        expected = expected.loc[:, ['A', 'B']]
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug2(self):
        df = DataFrame({'A': [0, 0, 0, 1, 1, 1], 'B': [1, 1, 2, 2, 3, 3], 'C': [1, 2, 3, 4, 5, 6]})
        expected = df.groupby('A').sum()
        g = df.groupby([Grouper(key='A')])
        result = g.sum()
        tm.assert_frame_equal(result, expected)
        expected = df.groupby(['A', 'B']).sum()
        g = df.groupby([Grouper(key='A'), Grouper(key='B')])
        result = g.sum()
        tm.assert_frame_equal(result, expected)
        g = df.groupby(['A', Grouper(key='B')])
        result = g.sum()
        tm.assert_frame_equal(result, expected)
        g = df.groupby([Grouper(key='A'), 'B'])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug3(self, unit):
        dti = date_range('20130101', periods=2, unit=unit)
        mi = MultiIndex.from_product([list('ab'), range(2), dti], names=['one', 'two', 'three'])
        ser = Series(np.arange(8, dtype='int64'), index=mi)
        result = ser.groupby(Grouper(level='three', freq='ME')).sum()
        exp_dti = pd.DatetimeIndex([Timestamp('2013-01-31')], freq='ME', name='three').as_unit(unit)
        expected = Series([28], index=exp_dti)
        tm.assert_series_equal(result, expected)
        result = ser.groupby(Grouper(level='one')).sum()
        expected = ser.groupby(level='one').sum()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func', [False, True])
    def test_grouper_returning_tuples(self, func):
        df = DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 4, 3, 2]})
        mapping = dict(zip(range(4), [('C', 5), ('D', 6)] * 2))
        if func:
            gb = df.groupby(by=lambda idx: mapping[idx], sort=False)
        else:
            gb = df.groupby(by=mapping, sort=False)
        name, expected = next(iter(gb))
        assert name == ('C', 5)
        result = gb.get_group(name)
        tm.assert_frame_equal(result, expected)

    def test_grouper_column_and_index(self):
        idx = MultiIndex.from_tuples([('a', 1), ('a', 2), ('a', 3), ('b', 1), ('b', 2), ('b', 3)])
        idx.names = ['outer', 'inner']
        df_multi = DataFrame({'A': np.arange(6), 'B': ['one', 'one', 'two', 'two', 'one', 'one']}, index=idx)
        result = df_multi.groupby(['B', Grouper(level='inner')]).mean(numeric_only=True)
        expected = df_multi.reset_index().groupby(['B', 'inner']).mean(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        result = df_multi.groupby([Grouper(level='inner'), 'B']).mean(numeric_only=True)
        expected = df_multi.reset_index().groupby(['inner', 'B']).mean(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        df_single = df_multi.reset_index('outer')
        result = df_single.groupby(['B', Grouper(level='inner')]).mean(numeric_only=True)
        expected = df_single.reset_index().groupby(['B', 'inner']).mean(numeric_only=True)
        tm.assert_frame_equal(result, expected)
        result = df_single.groupby([Grouper(level='inner'), 'B']).mean(numeric_only=True)
        expected = df_single.reset_index().groupby(['inner', 'B']).mean(numeric_only=True)
        tm.assert_frame_equal(result, expected)

    def test_groupby_levels_and_columns(self):
        idx_names = ['x', 'y']
        idx = MultiIndex.from_tuples([(1, 1), (1, 2), (3, 4), (5, 6)], names=idx_names)
        df = DataFrame(np.arange(12).reshape(-1, 3), index=idx)
        by_levels = df.groupby(level=idx_names).mean()
        by_columns = df.reset_index().groupby(idx_names).mean()
        by_columns.columns = by_columns.columns.astype(np.int64)
        tm.assert_frame_equal(by_levels, by_columns)

    def test_groupby_categorical_index_and_columns(self, observed):
        columns = ['A', 'B', 'A', 'B']
        categories = ['B', 'A']
        data = np.array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]], int)
        cat_columns = CategoricalIndex(columns, categories=categories, ordered=True)
        df = DataFrame(data=data, columns=cat_columns)
        depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = df.groupby(axis=1, level=0, observed=observed).sum()
        expected_data = np.array([[4, 2], [4, 2], [4, 2], [4, 2], [4, 2]], int)
        expected_columns = CategoricalIndex(categories, categories=categories, ordered=True)
        expected = DataFrame(data=expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)
        df = DataFrame(data.T, index=cat_columns)
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.groupby(axis=0, level=0, observed=observed).sum()
        expected = DataFrame(data=expected_data.T, index=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_grouper_getting_correct_binner(self):
        df = DataFrame({'A': 1}, index=MultiIndex.from_product([list('ab'), date_range('20130101', periods=80)], names=['one', 'two']))
        result = df.groupby([Grouper(level='one'), Grouper(level='two', freq='ME')]).sum()
        expected = DataFrame({'A': [31, 28, 21, 31, 28, 21]}, index=MultiIndex.from_product([list('ab'), date_range('20130101', freq='ME', periods=3)], names=['one', 'two']))
        tm.assert_frame_equal(result, expected)

    def test_grouper_iter(self, df):
        gb = df.groupby('A')
        msg = 'DataFrameGroupBy.grouper is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouper = gb.grouper
        result = sorted(grouper)
        expected = ['bar', 'foo']
        assert result == expected

    def test_empty_groups(self, df):
        with pytest.raises(ValueError, match='No group keys passed!'):
            df.groupby([])

    def test_groupby_grouper(self, df):
        grouped = df.groupby('A')
        msg = 'DataFrameGroupBy.grouper is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouper = grouped.grouper
        result = df.groupby(grouper).mean(numeric_only=True)
        expected = grouped.mean(numeric_only=True)
        tm.assert_frame_equal(result, expected)

    def test_groupby_dict_mapping(self):
        s = Series({'T1': 5})
        result = s.groupby({'T1': 'T2'}).agg('sum')
        expected = s.groupby(['T2']).agg('sum')
        tm.assert_series_equal(result, expected)
        s = Series([1.0, 2.0, 3.0, 4.0], index=list('abcd'))
        mapping = {'a': 0, 'b': 0, 'c': 1, 'd': 1}
        result = s.groupby(mapping).mean()
        result2 = s.groupby(mapping).agg('mean')
        exp_key = np.array([0, 0, 1, 1], dtype=np.int64)
        expected = s.groupby(exp_key).mean()
        expected2 = s.groupby(exp_key).mean()
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result, result2)
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('index', [[0, 1, 2, 3], ['a', 'b', 'c', 'd'], [Timestamp(2021, 7, 28 + i) for i in range(4)]])
    def test_groupby_series_named_with_tuple(self, frame_or_series, index):
        obj = frame_or_series([1, 2, 3, 4], index=index)
        groups = Series([1, 0, 1, 0], index=index, name=('a', 'a'))
        result = obj.groupby(groups).last()
        expected = frame_or_series([4, 3])
        expected.index.name = ('a', 'a')
        tm.assert_equal(result, expected)

    def test_groupby_grouper_f_sanity_checked(self):
        dates = date_range('01-Jan-2013', periods=12, freq='MS')
        ts = Series(np.random.default_rng(2).standard_normal(12), index=dates)
        msg = "'Timestamp' object is not subscriptable"
        with pytest.raises(TypeError, match=msg):
            ts.groupby(lambda key: key[0:6])
        result = ts.groupby(lambda x: x).sum()
        expected = ts.groupby(ts.index).sum()
        expected.index.freq = None
        tm.assert_series_equal(result, expected)

    def test_groupby_with_datetime_key(self):
        df = DataFrame({'id': ['a', 'b'] * 3, 'b': date_range('2000-01-01', '2000-01-03', freq='9h')})
        grouper = Grouper(key='b', freq='D')
        gb = df.groupby([grouper, 'id'])
        expected = {(Timestamp('2000-01-01'), 'a'): [0, 2], (Timestamp('2000-01-01'), 'b'): [1], (Timestamp('2000-01-02'), 'a'): [4], (Timestamp('2000-01-02'), 'b'): [3, 5]}
        tm.assert_dict_equal(gb.groups, expected)
        assert len(gb.groups.keys()) == 4

    def test_grouping_error_on_multidim_input(self, df):
        msg = "Grouper for '<class 'pandas.core.frame.DataFrame'>' not 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            Grouping(df.index, df[['A', 'A']])

    def test_multiindex_passthru(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        df.columns = MultiIndex.from_tuples([(0, 1), (1, 1), (2, 1)])
        depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            gb = df.groupby(axis=1, level=[0, 1])
        result = gb.first()
        tm.assert_frame_equal(result, df)

    def test_multiindex_negative_level(self, multiindex_dataframe_random_data):
        result = multiindex_dataframe_random_data.groupby(level=-1).sum()
        expected = multiindex_dataframe_random_data.groupby(level='second').sum()
        tm.assert_frame_equal(result, expected)
        result = multiindex_dataframe_random_data.groupby(level=-2).sum()
        expected = multiindex_dataframe_random_data.groupby(level='first').sum()
        tm.assert_frame_equal(result, expected)
        result = multiindex_dataframe_random_data.groupby(level=[-2, -1]).sum()
        expected = multiindex_dataframe_random_data.sort_index()
        tm.assert_frame_equal(result, expected)
        result = multiindex_dataframe_random_data.groupby(level=[-1, 'first']).sum()
        expected = multiindex_dataframe_random_data.groupby(level=['second', 'first']).sum()
        tm.assert_frame_equal(result, expected)

    def test_multifunc_select_col_integer_cols(self, df):
        df.columns = np.arange(len(df.columns))
        msg = 'Passing a dictionary to SeriesGroupBy.agg is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.groupby(1, as_index=False)[2].agg({'Q': np.mean})

    def test_multiindex_columns_empty_level(self):
        lst = [['count', 'values'], ['to filter', '']]
        midx = MultiIndex.from_tuples(lst)
        df = DataFrame([[1, 'A']], columns=midx)
        grouped = df.groupby('to filter').groups
        assert grouped['A'] == [0]
        grouped = df.groupby([('to filter', '')]).groups
        assert grouped['A'] == [0]
        df = DataFrame([[1, 'A'], [2, 'B']], columns=midx)
        expected = df.groupby('to filter').groups
        result = df.groupby([('to filter', '')]).groups
        assert result == expected
        df = DataFrame([[1, 'A'], [2, 'A']], columns=midx)
        expected = df.groupby('to filter').groups
        result = df.groupby([('to filter', '')]).groups
        tm.assert_dict_equal(result, expected)

    def test_groupby_multiindex_tuple(self):
        df = DataFrame([[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]], columns=MultiIndex.from_arrays([['a', 'b', 'b', 'c'], [1, 1, 2, 2]]))
        expected = df.groupby([('b', 1)]).groups
        result = df.groupby(('b', 1)).groups
        tm.assert_dict_equal(expected, result)
        df2 = DataFrame(df.values, columns=MultiIndex.from_arrays([['a', 'b', 'b', 'c'], ['d', 'd', 'e', 'e']]))
        expected = df2.groupby([('b', 'd')]).groups
        result = df.groupby(('b', 1)).groups
        tm.assert_dict_equal(expected, result)
        df3 = DataFrame(df.values, columns=[('a', 'd'), ('b', 'd'), ('b', 'e'), 'c'])
        expected = df3.groupby([('b', 'd')]).groups
        result = df.groupby(('b', 1)).groups
        tm.assert_dict_equal(expected, result)

    def test_groupby_multiindex_partial_indexing_equivalence(self):
        df = DataFrame([[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]], columns=MultiIndex.from_arrays([['a', 'b', 'b', 'c'], [1, 1, 2, 2]]))
        expected_mean = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].mean()
        result_mean = df.groupby([('a', 1)])['b'].mean()
        tm.assert_frame_equal(expected_mean, result_mean)
        expected_sum = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].sum()
        result_sum = df.groupby([('a', 1)])['b'].sum()
        tm.assert_frame_equal(expected_sum, result_sum)
        expected_count = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].count()
        result_count = df.groupby([('a', 1)])['b'].count()
        tm.assert_frame_equal(expected_count, result_count)
        expected_min = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].min()
        result_min = df.groupby([('a', 1)])['b'].min()
        tm.assert_frame_equal(expected_min, result_min)
        expected_max = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].max()
        result_max = df.groupby([('a', 1)])['b'].max()
        tm.assert_frame_equal(expected_max, result_max)
        expected_groups = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].groups
        result_groups = df.groupby([('a', 1)])['b'].groups
        tm.assert_dict_equal(expected_groups, result_groups)

    @pytest.mark.parametrize('sort', [True, False])
    def test_groupby_level(self, sort, multiindex_dataframe_random_data, df):
        frame = multiindex_dataframe_random_data
        deleveled = frame.reset_index()
        result0 = frame.groupby(level=0, sort=sort).sum()
        result1 = frame.groupby(level=1, sort=sort).sum()
        expected0 = frame.groupby(deleveled['first'].values, sort=sort).sum()
        expected1 = frame.groupby(deleveled['second'].values, sort=sort).sum()
        expected0.index.name = 'first'
        expected1.index.name = 'second'
        assert result0.index.name == 'first'
        assert result1.index.name == 'second'
        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)
        assert result0.index.name == frame.index.names[0]
        assert result1.index.name == frame.index.names[1]
        result0 = frame.groupby(level='first', sort=sort).sum()
        result1 = frame.groupby(level='second', sort=sort).sum()
        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)
        msg = 'DataFrame.groupby with axis=1 is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result0 = frame.T.groupby(level=0, axis=1, sort=sort).sum()
            result1 = frame.T.groupby(level=1, axis=1, sort=sort).sum()
        tm.assert_frame_equal(result0, expected0.T)
        tm.assert_frame_equal(result1, expected1.T)
        msg = 'level > 0 or level < -1 only valid with MultiIndex'
        with pytest.raises(ValueError, match=msg):
            df.groupby(level=1)

    def test_groupby_level_index_names(self, axis):
        df = DataFrame({'exp': ['A'] * 3 + ['B'] * 3, 'var1': range(6)}).set_index('exp')
        if axis in (1, 'columns'):
            df = df.T
            depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
        else:
            depr_msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df.groupby(level='exp', axis=axis)
        msg = f'level name foo is not the name of the {df._get_axis_name(axis)}'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                df.groupby(level='foo', axis=axis)

    @pytest.mark.parametrize('sort', [True, False])
    def test_groupby_level_with_nas(self, sort):
        index = MultiIndex(levels=[[1, 0], [0, 1, 2, 3]], codes=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]])
        s = Series(np.arange(8.0), index=index)
        result = s.groupby(level=0, sort=sort).sum()
        expected = Series([6.0, 22.0], index=[0, 1])
        tm.assert_series_equal(result, expected)
        index = MultiIndex(levels=[[1, 0], [0, 1, 2, 3]], codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]])
        s = Series(np.arange(8.0), index=index)
        result = s.groupby(level=0, sort=sort).sum()
        expected = Series([6.0, 18.0], index=[0.0, 1.0])
        tm.assert_series_equal(result, expected)

    def test_groupby_args(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        msg = "You have to supply one of 'by' and 'level'"
        with pytest.raises(TypeError, match=msg):
            frame.groupby()
        msg = "You have to supply one of 'by' and 'level'"
        with pytest.raises(TypeError, match=msg):
            frame.groupby(by=None, level=None)

    @pytest.mark.parametrize('sort,labels', [[True, [2, 2, 2, 0, 0, 1, 1, 3, 3, 3]], [False, [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]]])
    def test_level_preserve_order(self, sort, labels, multiindex_dataframe_random_data):
        grouped = multiindex_dataframe_random_data.groupby(level=0, sort=sort)
        exp_labels = np.array(labels, np.intp)
        tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)

    def test_grouping_labels(self, multiindex_dataframe_random_data):
        grouped = multiindex_dataframe_random_data.groupby(multiindex_dataframe_random_data.index.get_level_values(0))
        exp_labels = np.array([2, 2, 2, 0, 0, 1, 1, 3, 3, 3], dtype=np.intp)
        tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)

    def test_list_grouper_with_nat(self):
        df = DataFrame({'date': date_range('1/1/2011', periods=365, freq='D')})
        df.iloc[-1] = pd.NaT
        grouper = Grouper(key='date', freq='YS')
        result = df.groupby([grouper])
        expected = {Timestamp('2011-01-01'): Index(list(range(364)))}
        tm.assert_dict_equal(result.groups, expected)
        result = df.groupby(grouper)
        expected = {Timestamp('2011-01-01'): 365}
        tm.assert_dict_equal(result.groups, expected)

    @pytest.mark.parametrize('func,expected', [('transform', Series(name=2, dtype=np.float64)), ('agg', Series(name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1))), ('apply', Series(name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1)))])
    def test_evaluate_with_empty_groups(self, func, expected):
        df = DataFrame({1: [], 2: []})
        g = df.groupby(1, group_keys=False)
        result = getattr(g[2], func)(lambda x: x)
        tm.assert_series_equal(result, expected)

    def test_groupby_empty(self):
        s = Series([], name='name', dtype='float64')
        gr = s.groupby([])
        result = gr.mean()
        expected = s.set_axis(Index([], dtype=np.intp))
        tm.assert_series_equal(result, expected)
        assert len(gr._grouper.groupings) == 1
        tm.assert_numpy_array_equal(gr._grouper.group_info[0], np.array([], dtype=np.dtype(np.intp)))
        tm.assert_numpy_array_equal(gr._grouper.group_info[1], np.array([], dtype=np.dtype(np.intp)))
        assert gr._grouper.group_info[2] == 0
        gb = s.groupby(s)
        msg = 'SeriesGroupBy.grouper is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouper = gb.grouper
        result = grouper.names
        expected = ['name']
        assert result == expected

    def test_groupby_level_index_value_all_na(self):
        df = DataFrame([['x', np.nan, 10], [None, np.nan, 20]], columns=['A', 'B', 'C']).set_index(['A', 'B'])
        result = df.groupby(level=['A', 'B']).sum()
        expected = DataFrame(data=[], index=MultiIndex(levels=[Index(['x'], dtype='object'), Index([], dtype='float64')], codes=[[], []], names=['A', 'B']), columns=['C'], dtype='int64')
        tm.assert_frame_equal(result, expected)

    def test_groupby_multiindex_level_empty(self):
        df = DataFrame([[123, 'a', 1.0], [123, 'b', 2.0]], columns=['id', 'category', 'value'])
        df = df.set_index(['id', 'category'])
        empty = df[df.value < 0]
        result = empty.groupby('id').sum()
        expected = DataFrame(dtype='float64', columns=['value'], index=Index([], dtype=np.int64, name='id'))
        tm.assert_frame_equal(result, expected)