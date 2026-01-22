from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
class TestPivot:

    def test_pivot(self):
        data = {'index': ['A', 'B', 'C', 'C', 'B', 'A'], 'columns': ['One', 'One', 'One', 'Two', 'Two', 'Two'], 'values': [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]}
        frame = DataFrame(data)
        pivoted = frame.pivot(index='index', columns='columns', values='values')
        expected = DataFrame({'One': {'A': 1.0, 'B': 2.0, 'C': 3.0}, 'Two': {'A': 1.0, 'B': 2.0, 'C': 3.0}})
        expected.index.name, expected.columns.name = ('index', 'columns')
        tm.assert_frame_equal(pivoted, expected)
        assert pivoted.index.name == 'index'
        assert pivoted.columns.name == 'columns'
        pivoted = frame.pivot(index='index', columns='columns')
        assert pivoted.index.name == 'index'
        assert pivoted.columns.names == (None, 'columns')

    def test_pivot_duplicates(self):
        data = DataFrame({'a': ['bar', 'bar', 'foo', 'foo', 'foo'], 'b': ['one', 'two', 'one', 'one', 'two'], 'c': [1.0, 2.0, 3.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match='duplicate entries'):
            data.pivot(index='a', columns='b', values='c')

    def test_pivot_empty(self):
        df = DataFrame(columns=['a', 'b', 'c'])
        result = df.pivot(index='a', columns='b', values='c')
        expected = DataFrame(index=[], columns=[])
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.parametrize('dtype', [object, 'string'])
    def test_pivot_integer_bug(self, dtype):
        df = DataFrame(data=[('A', '1', 'A1'), ('B', '2', 'B2')], dtype=dtype)
        result = df.pivot(index=1, columns=0, values=2)
        tm.assert_index_equal(result.columns, Index(['A', 'B'], name=0, dtype=dtype))

    def test_pivot_index_none(self):
        data = {'index': ['A', 'B', 'C', 'C', 'B', 'A'], 'columns': ['One', 'One', 'One', 'Two', 'Two', 'Two'], 'values': [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]}
        frame = DataFrame(data).set_index('index')
        result = frame.pivot(columns='columns', values='values')
        expected = DataFrame({'One': {'A': 1.0, 'B': 2.0, 'C': 3.0}, 'Two': {'A': 1.0, 'B': 2.0, 'C': 3.0}})
        expected.index.name, expected.columns.name = ('index', 'columns')
        tm.assert_frame_equal(result, expected)
        result = frame.pivot(columns='columns')
        expected.columns = MultiIndex.from_tuples([('values', 'One'), ('values', 'Two')], names=[None, 'columns'])
        expected.index.name = 'index'
        tm.assert_frame_equal(result, expected, check_names=False)
        assert result.index.name == 'index'
        assert result.columns.names == (None, 'columns')
        expected.columns = expected.columns.droplevel(0)
        result = frame.pivot(columns='columns', values='values')
        expected.columns.name = 'columns'
        tm.assert_frame_equal(result, expected)

    def test_pivot_index_list_values_none_immutable_args(self):
        df = DataFrame({'lev1': [1, 1, 1, 2, 2, 2], 'lev2': [1, 1, 2, 1, 1, 2], 'lev3': [1, 2, 1, 2, 1, 2], 'lev4': [1, 2, 3, 4, 5, 6], 'values': [0, 1, 2, 3, 4, 5]})
        index = ['lev1', 'lev2']
        columns = ['lev3']
        result = df.pivot(index=index, columns=columns)
        expected = DataFrame(np.array([[1.0, 2.0, 0.0, 1.0], [3.0, np.nan, 2.0, np.nan], [5.0, 4.0, 4.0, 3.0], [np.nan, 6.0, np.nan, 5.0]]), index=MultiIndex.from_arrays([(1, 1, 2, 2), (1, 2, 1, 2)], names=['lev1', 'lev2']), columns=MultiIndex.from_arrays([('lev4', 'lev4', 'values', 'values'), (1, 2, 1, 2)], names=[None, 'lev3']))
        tm.assert_frame_equal(result, expected)
        assert index == ['lev1', 'lev2']
        assert columns == ['lev3']

    def test_pivot_columns_not_given(self):
        df = DataFrame({'a': [1], 'b': 1})
        with pytest.raises(TypeError, match='missing 1 required keyword-only argument'):
            df.pivot()

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason='None is cast to NaN')
    def test_pivot_columns_is_none(self):
        df = DataFrame({None: [1], 'b': 2, 'c': 3})
        result = df.pivot(columns=None)
        expected = DataFrame({('b', 1): [2], ('c', 1): 3})
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns=None, index='b')
        expected = DataFrame({('c', 1): 3}, index=Index([2], name='b'))
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns=None, index='b', values='c')
        expected = DataFrame({1: 3}, index=Index([2], name='b'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason='None is cast to NaN')
    def test_pivot_index_is_none(self):
        df = DataFrame({None: [1], 'b': 2, 'c': 3})
        result = df.pivot(columns='b', index=None)
        expected = DataFrame({('c', 2): 3}, index=[1])
        expected.columns.names = [None, 'b']
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns='b', index=None, values='c')
        expected = DataFrame(3, index=[1], columns=Index([2], name='b'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason='None is cast to NaN')
    def test_pivot_values_is_none(self):
        df = DataFrame({None: [1], 'b': 2, 'c': 3})
        result = df.pivot(columns='b', index='c', values=None)
        expected = DataFrame(1, index=Index([3], name='c'), columns=Index([2], name='b'))
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns='b', values=None)
        expected = DataFrame(1, index=[0], columns=Index([2], name='b'))
        tm.assert_frame_equal(result, expected)

    def test_pivot_not_changing_index_name(self):
        df = DataFrame({'one': ['a'], 'two': 0, 'three': 1})
        expected = df.copy(deep=True)
        df.pivot(index='one', columns='two', values='three')
        tm.assert_frame_equal(df, expected)

    def test_pivot_table_empty_dataframe_correct_index(self):
        df = DataFrame([], columns=['a', 'b', 'value'])
        pivot = df.pivot_table(index='a', columns='b', values='value', aggfunc='count')
        expected = Index([], dtype='object', name='b')
        tm.assert_index_equal(pivot.columns, expected)

    def test_pivot_table_handles_explicit_datetime_types(self):
        df = DataFrame([{'a': 'x', 'date_str': '2023-01-01', 'amount': 1}, {'a': 'y', 'date_str': '2023-01-02', 'amount': 2}, {'a': 'z', 'date_str': '2023-01-03', 'amount': 3}])
        df['date'] = pd.to_datetime(df['date_str'])
        with tm.assert_produces_warning(False):
            pivot = df.pivot_table(index=['a', 'date'], values=['amount'], aggfunc='sum', margins=True)
        expected = MultiIndex.from_tuples([('x', datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')), ('y', datetime.strptime('2023-01-02 00:00:00', '%Y-%m-%d %H:%M:%S')), ('z', datetime.strptime('2023-01-03 00:00:00', '%Y-%m-%d %H:%M:%S')), ('All', '')], names=['a', 'date'])
        tm.assert_index_equal(pivot.index, expected)

    def test_pivot_table_with_margins_and_numeric_column_names(self):
        df = DataFrame([['a', 'x', 1], ['a', 'y', 2], ['b', 'y', 3], ['b', 'z', 4]])
        result = df.pivot_table(index=0, columns=1, values=2, aggfunc='sum', fill_value=0, margins=True)
        expected = DataFrame([[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]], columns=Index(['x', 'y', 'z', 'All'], name=1), index=Index(['a', 'b', 'All'], name=0))
        tm.assert_frame_equal(result, expected)