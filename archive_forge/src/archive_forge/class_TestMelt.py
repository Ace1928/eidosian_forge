import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestMelt:

    def test_top_level_method(self, df):
        result = melt(df)
        assert result.columns.tolist() == ['variable', 'value']

    def test_method_signatures(self, df, df1, var_name, value_name):
        tm.assert_frame_equal(df.melt(), melt(df))
        tm.assert_frame_equal(df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B']), melt(df, id_vars=['id1', 'id2'], value_vars=['A', 'B']))
        tm.assert_frame_equal(df.melt(var_name=var_name, value_name=value_name), melt(df, var_name=var_name, value_name=value_name))
        tm.assert_frame_equal(df1.melt(col_level=0), melt(df1, col_level=0))

    def test_default_col_names(self, df):
        result = df.melt()
        assert result.columns.tolist() == ['variable', 'value']
        result1 = df.melt(id_vars=['id1'])
        assert result1.columns.tolist() == ['id1', 'variable', 'value']
        result2 = df.melt(id_vars=['id1', 'id2'])
        assert result2.columns.tolist() == ['id1', 'id2', 'variable', 'value']

    def test_value_vars(self, df):
        result3 = df.melt(id_vars=['id1', 'id2'], value_vars='A')
        assert len(result3) == 10
        result4 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'])
        expected4 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, 'variable': ['A'] * 10 + ['B'] * 10, 'value': df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', 'variable', 'value'])
        tm.assert_frame_equal(result4, expected4)

    @pytest.mark.parametrize('type_', (tuple, list, np.array))
    def test_value_vars_types(self, type_, df):
        expected = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, 'variable': ['A'] * 10 + ['B'] * 10, 'value': df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', 'variable', 'value'])
        result = df.melt(id_vars=['id1', 'id2'], value_vars=type_(('A', 'B')))
        tm.assert_frame_equal(result, expected)

    def test_vars_work_with_multiindex(self, df1):
        expected = DataFrame({('A', 'a'): df1['A', 'a'], 'CAP': ['B'] * len(df1), 'low': ['b'] * len(df1), 'value': df1['B', 'b']}, columns=[('A', 'a'), 'CAP', 'low', 'value'])
        result = df1.melt(id_vars=[('A', 'a')], value_vars=[('B', 'b')])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('id_vars, value_vars, col_level, expected', [(['A'], ['B'], 0, DataFrame({'A': {0: 1.067683, 1: -1.321405, 2: -0.807333}, 'CAP': {0: 'B', 1: 'B', 2: 'B'}, 'value': {0: -1.110463, 1: 0.368915, 2: 0.08298}})), (['a'], ['b'], 1, DataFrame({'a': {0: 1.067683, 1: -1.321405, 2: -0.807333}, 'low': {0: 'b', 1: 'b', 2: 'b'}, 'value': {0: -1.110463, 1: 0.368915, 2: 0.08298}}))])
    def test_single_vars_work_with_multiindex(self, id_vars, value_vars, col_level, expected, df1):
        result = df1.melt(id_vars, value_vars, col_level=col_level)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('id_vars, value_vars', [[('A', 'a'), [('B', 'b')]], [[('A', 'a')], ('B', 'b')], [('A', 'a'), ('B', 'b')]])
    def test_tuple_vars_fail_with_multiindex(self, id_vars, value_vars, df1):
        msg = '(id|value)_vars must be a list of tuples when columns are a MultiIndex'
        with pytest.raises(ValueError, match=msg):
            df1.melt(id_vars=id_vars, value_vars=value_vars)

    def test_custom_var_name(self, df, var_name):
        result5 = df.melt(var_name=var_name)
        assert result5.columns.tolist() == ['var', 'value']
        result6 = df.melt(id_vars=['id1'], var_name=var_name)
        assert result6.columns.tolist() == ['id1', 'var', 'value']
        result7 = df.melt(id_vars=['id1', 'id2'], var_name=var_name)
        assert result7.columns.tolist() == ['id1', 'id2', 'var', 'value']
        result8 = df.melt(id_vars=['id1', 'id2'], value_vars='A', var_name=var_name)
        assert result8.columns.tolist() == ['id1', 'id2', 'var', 'value']
        result9 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'], var_name=var_name)
        expected9 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, var_name: ['A'] * 10 + ['B'] * 10, 'value': df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', var_name, 'value'])
        tm.assert_frame_equal(result9, expected9)

    def test_custom_value_name(self, df, value_name):
        result10 = df.melt(value_name=value_name)
        assert result10.columns.tolist() == ['variable', 'val']
        result11 = df.melt(id_vars=['id1'], value_name=value_name)
        assert result11.columns.tolist() == ['id1', 'variable', 'val']
        result12 = df.melt(id_vars=['id1', 'id2'], value_name=value_name)
        assert result12.columns.tolist() == ['id1', 'id2', 'variable', 'val']
        result13 = df.melt(id_vars=['id1', 'id2'], value_vars='A', value_name=value_name)
        assert result13.columns.tolist() == ['id1', 'id2', 'variable', 'val']
        result14 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'], value_name=value_name)
        expected14 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, 'variable': ['A'] * 10 + ['B'] * 10, value_name: df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', 'variable', value_name])
        tm.assert_frame_equal(result14, expected14)

    def test_custom_var_and_value_name(self, df, value_name, var_name):
        result15 = df.melt(var_name=var_name, value_name=value_name)
        assert result15.columns.tolist() == ['var', 'val']
        result16 = df.melt(id_vars=['id1'], var_name=var_name, value_name=value_name)
        assert result16.columns.tolist() == ['id1', 'var', 'val']
        result17 = df.melt(id_vars=['id1', 'id2'], var_name=var_name, value_name=value_name)
        assert result17.columns.tolist() == ['id1', 'id2', 'var', 'val']
        result18 = df.melt(id_vars=['id1', 'id2'], value_vars='A', var_name=var_name, value_name=value_name)
        assert result18.columns.tolist() == ['id1', 'id2', 'var', 'val']
        result19 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'], var_name=var_name, value_name=value_name)
        expected19 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, var_name: ['A'] * 10 + ['B'] * 10, value_name: df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', var_name, value_name])
        tm.assert_frame_equal(result19, expected19)
        df20 = df.copy()
        df20.columns.name = 'foo'
        result20 = df20.melt()
        assert result20.columns.tolist() == ['foo', 'value']

    @pytest.mark.parametrize('col_level', [0, 'CAP'])
    def test_col_level(self, col_level, df1):
        res = df1.melt(col_level=col_level)
        assert res.columns.tolist() == ['CAP', 'value']

    def test_multiindex(self, df1):
        res = df1.melt()
        assert res.columns.tolist() == ['CAP', 'low', 'value']

    @pytest.mark.parametrize('col', [pd.Series(date_range('2010', periods=5, tz='US/Pacific')), pd.Series(['a', 'b', 'c', 'a', 'd'], dtype='category'), pd.Series([0, 1, 0, 0, 0])])
    def test_pandas_dtypes(self, col):
        df = DataFrame({'klass': range(5), 'col': col, 'attr1': [1, 0, 0, 0, 0], 'attr2': col})
        expected_value = pd.concat([pd.Series([1, 0, 0, 0, 0]), col], ignore_index=True)
        result = melt(df, id_vars=['klass', 'col'], var_name='attribute', value_name='value')
        expected = DataFrame({0: list(range(5)) * 2, 1: pd.concat([col] * 2, ignore_index=True), 2: ['attr1'] * 5 + ['attr2'] * 5, 3: expected_value})
        expected.columns = ['klass', 'col', 'attribute', 'value']
        tm.assert_frame_equal(result, expected)

    def test_preserve_category(self):
        data = DataFrame({'A': [1, 2], 'B': pd.Categorical(['X', 'Y'])})
        result = melt(data, ['B'], ['A'])
        expected = DataFrame({'B': pd.Categorical(['X', 'Y']), 'variable': ['A', 'A'], 'value': [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_melt_missing_columns_raises(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=list('abcd'))
        msg = 'The following id_vars or value_vars are not present in the DataFrame:'
        with pytest.raises(KeyError, match=msg):
            df.melt(['a', 'b'], ['C', 'd'])
        with pytest.raises(KeyError, match=msg):
            df.melt(['A', 'b'], ['c', 'd'])
        with pytest.raises(KeyError, match=msg):
            df.melt(['a', 'b', 'not_here', 'or_there'], ['c', 'd'])
        multi = df.copy()
        multi.columns = [list('ABCD'), list('abcd')]
        with pytest.raises(KeyError, match=msg):
            multi.melt([('E', 'a')], [('B', 'b')])
        with pytest.raises(KeyError, match=msg):
            multi.melt(['A'], ['F'], col_level=0)

    def test_melt_mixed_int_str_id_vars(self):
        df = DataFrame({0: ['foo'], 'a': ['bar'], 'b': [1], 'd': [2]})
        result = melt(df, id_vars=[0, 'a'], value_vars=['b', 'd'])
        expected = DataFrame({0: ['foo'] * 2, 'a': ['bar'] * 2, 'variable': list('bd'), 'value': [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_melt_mixed_int_str_value_vars(self):
        df = DataFrame({0: ['foo'], 'a': ['bar']})
        result = melt(df, value_vars=[0, 'a'])
        expected = DataFrame({'variable': [0, 'a'], 'value': ['foo', 'bar']})
        tm.assert_frame_equal(result, expected)

    def test_ignore_index(self):
        df = DataFrame({'foo': [0], 'bar': [1]}, index=['first'])
        result = melt(df, ignore_index=False)
        expected = DataFrame({'variable': ['foo', 'bar'], 'value': [0, 1]}, index=['first', 'first'])
        tm.assert_frame_equal(result, expected)

    def test_ignore_multiindex(self):
        index = pd.MultiIndex.from_tuples([('first', 'second'), ('first', 'third')], names=['baz', 'foobar'])
        df = DataFrame({'foo': [0, 1], 'bar': [2, 3]}, index=index)
        result = melt(df, ignore_index=False)
        expected_index = pd.MultiIndex.from_tuples([('first', 'second'), ('first', 'third')] * 2, names=['baz', 'foobar'])
        expected = DataFrame({'variable': ['foo'] * 2 + ['bar'] * 2, 'value': [0, 1, 2, 3]}, index=expected_index)
        tm.assert_frame_equal(result, expected)

    def test_ignore_index_name_and_type(self):
        index = Index(['foo', 'bar'], dtype='category', name='baz')
        df = DataFrame({'x': [0, 1], 'y': [2, 3]}, index=index)
        result = melt(df, ignore_index=False)
        expected_index = Index(['foo', 'bar'] * 2, dtype='category', name='baz')
        expected = DataFrame({'variable': ['x', 'x', 'y', 'y'], 'value': [0, 1, 2, 3]}, index=expected_index)
        tm.assert_frame_equal(result, expected)

    def test_melt_with_duplicate_columns(self):
        df = DataFrame([['id', 2, 3]], columns=['a', 'b', 'b'])
        result = df.melt(id_vars=['a'], value_vars=['b'])
        expected = DataFrame([['id', 'b', 2], ['id', 'b', 3]], columns=['a', 'variable', 'value'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['Int8', 'Int64'])
    def test_melt_ea_dtype(self, dtype):
        df = DataFrame({'a': pd.Series([1, 2], dtype='Int8'), 'b': pd.Series([3, 4], dtype=dtype)})
        result = df.melt()
        expected = DataFrame({'variable': ['a', 'a', 'b', 'b'], 'value': pd.Series([1, 2, 3, 4], dtype=dtype)})
        tm.assert_frame_equal(result, expected)

    def test_melt_ea_columns(self):
        df = DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'}, 'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}})
        df.columns = df.columns.astype('string[python]')
        result = df.melt(id_vars=['A'], value_vars=['B'])
        expected = DataFrame({'A': list('abc'), 'variable': pd.Series(['B'] * 3, dtype='string[python]'), 'value': [1, 3, 5]})
        tm.assert_frame_equal(result, expected)

    def test_melt_preserves_datetime(self):
        df = DataFrame(data=[{'type': 'A0', 'start_date': pd.Timestamp('2023/03/01', tz='Asia/Tokyo'), 'end_date': pd.Timestamp('2023/03/10', tz='Asia/Tokyo')}, {'type': 'A1', 'start_date': pd.Timestamp('2023/03/01', tz='Asia/Tokyo'), 'end_date': pd.Timestamp('2023/03/11', tz='Asia/Tokyo')}], index=['aaaa', 'bbbb'])
        result = df.melt(id_vars=['type'], value_vars=['start_date', 'end_date'], var_name='start/end', value_name='date')
        expected = DataFrame({'type': {0: 'A0', 1: 'A1', 2: 'A0', 3: 'A1'}, 'start/end': {0: 'start_date', 1: 'start_date', 2: 'end_date', 3: 'end_date'}, 'date': {0: pd.Timestamp('2023-03-01 00:00:00+0900', tz='Asia/Tokyo'), 1: pd.Timestamp('2023-03-01 00:00:00+0900', tz='Asia/Tokyo'), 2: pd.Timestamp('2023-03-10 00:00:00+0900', tz='Asia/Tokyo'), 3: pd.Timestamp('2023-03-11 00:00:00+0900', tz='Asia/Tokyo')}})
        tm.assert_frame_equal(result, expected)

    def test_melt_allows_non_scalar_id_vars(self):
        df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['11', '22', '33'])
        result = df.melt(id_vars='a', var_name=0, value_name=1)
        expected = DataFrame({'a': [1, 2, 3], 0: ['b'] * 3, 1: [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_melt_allows_non_string_var_name(self):
        df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['11', '22', '33'])
        result = df.melt(id_vars=['a'], var_name=0, value_name=1)
        expected = DataFrame({'a': [1, 2, 3], 0: ['b'] * 3, 1: [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_melt_non_scalar_var_name_raises(self):
        df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['11', '22', '33'])
        with pytest.raises(ValueError, match='.* must be a scalar.'):
            df.melt(id_vars=['a'], var_name=[1, 2])