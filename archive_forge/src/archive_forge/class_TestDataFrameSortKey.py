import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
class TestDataFrameSortKey:

    def test_sort_values_inplace_key(self, sort_by_key):
        frame = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by='A', inplace=True, key=sort_by_key)
        assert return_value is None
        expected = frame.sort_values(by='A', key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True, key=sort_by_key)
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by='A', ascending=False, inplace=True, key=sort_by_key)
        assert return_value is None
        expected = frame.sort_values(by='A', ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        sorted_df.sort_values(by=['A', 'B'], ascending=False, inplace=True, key=sort_by_key)
        expected = frame.sort_values(by=['A', 'B'], ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_key(self):
        df = DataFrame(np.array([0, 5, np.nan, 3, 2, np.nan]))
        result = df.sort_values(0)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(0, key=lambda x: x + 5)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(0, key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key(self):
        df = DataFrame({'a': np.array([0, 3, np.nan, 3, 2, np.nan]), 'b': np.array([0, 2, np.nan, 5, 2, np.nan])})
        result = df.sort_values('a', key=lambda x: -x)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a', 'b'], key=lambda x: -x)
        expected = df.iloc[[3, 1, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a', 'b'], key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 1, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key_by_name(self):
        df = DataFrame({'a': np.array([0, 3, np.nan, 3, 2, np.nan]), 'b': np.array([0, 2, np.nan, 5, 2, np.nan])})

        def key(col):
            if col.name == 'a':
                return -col
            else:
                return col
        result = df.sort_values(by='a', key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a'], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by='b', key=key)
        expected = df.iloc[[0, 1, 4, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a', 'b'], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_string(self):
        df = DataFrame(np.array([['hello', 'goodbye'], ['hello', 'Hello']]))
        result = df.sort_values(1)
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values([0, 1], key=lambda col: col.str.lower())
        tm.assert_frame_equal(result, df)
        result = df.sort_values([0, 1], key=lambda col: col.str.lower(), ascending=False)
        expected = df.sort_values(1, key=lambda col: col.str.lower(), ascending=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_empty(self, sort_by_key):
        df = DataFrame(np.array([]))
        df.sort_values(0, key=sort_by_key)
        df.sort_index(key=sort_by_key)

    def test_changes_length_raises(self):
        df = DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError, match='change the shape'):
            df.sort_values('A', key=lambda x: x[:1])

    def test_sort_values_key_axes(self):
        df = DataFrame({0: ['Hello', 'goodbye'], 1: [0, 1]})
        result = df.sort_values(0, key=lambda col: col.str.lower())
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(1, key=lambda col: -col)
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_dict_axis(self):
        df = DataFrame({0: ['Hello', 0], 1: ['goodbye', 1]})
        result = df.sort_values(0, key=lambda col: col.str.lower(), axis=1)
        expected = df.loc[:, ::-1]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(1, key=lambda col: -col, axis=1)
        expected = df.loc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [True, False])
    def test_sort_values_key_casts_to_categorical(self, ordered):
        categories = ['c', 'b', 'a']
        df = DataFrame({'x': [1, 1, 1], 'y': ['a', 'b', 'c']})

        def sorter(key):
            if key.name == 'y':
                return pd.Series(Categorical(key, categories=categories, ordered=ordered))
            return key
        result = df.sort_values(by=['x', 'y'], key=sorter)
        expected = DataFrame({'x': [1, 1, 1], 'y': ['c', 'b', 'a']}, index=pd.Index([2, 1, 0]))
        tm.assert_frame_equal(result, expected)