from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
class TestMergeCategorical:

    def test_identical(self, left, using_infer_string):
        merged = merge(left, left, on='X')
        result = merged.dtypes.sort_index()
        dtype = np.dtype('O') if not using_infer_string else 'string'
        expected = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, dtype], index=['X', 'Y_x', 'Y_y'])
        tm.assert_series_equal(result, expected)

    def test_basic(self, left, right, using_infer_string):
        merged = merge(left, right, on='X')
        result = merged.dtypes.sort_index()
        dtype = np.dtype('O') if not using_infer_string else 'string'
        expected = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
        tm.assert_series_equal(result, expected)

    def test_merge_categorical(self):
        right = DataFrame({'c': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}, 'd': {0: 'null', 1: 'null', 2: 'null', 3: 'null', 4: 'null'}})
        left = DataFrame({'a': {0: 'f', 1: 'f', 2: 'f', 3: 'f', 4: 'f'}, 'b': {0: 'g', 1: 'g', 2: 'g', 3: 'g', 4: 'g'}})
        df = merge(left, right, how='left', left_on='b', right_on='c')
        expected = df.copy()
        cright = right.copy()
        cright['d'] = cright['d'].astype('category')
        result = merge(left, cright, how='left', left_on='b', right_on='c')
        expected['d'] = expected['d'].astype(CategoricalDtype(['null']))
        tm.assert_frame_equal(result, expected)
        cleft = left.copy()
        cleft['b'] = cleft['b'].astype('category')
        result = merge(cleft, cright, how='left', left_on='b', right_on='c')
        tm.assert_frame_equal(result, expected)
        cright = right.copy()
        cright['d'] = cright['d'].astype('category')
        cleft = left.copy()
        cleft['b'] = cleft['b'].astype('category')
        result = merge(cleft, cright, how='left', left_on='b', right_on='c')
        tm.assert_frame_equal(result, expected)

    def tests_merge_categorical_unordered_equal(self):
        df1 = DataFrame({'Foo': Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']), 'Left': ['A0', 'B0', 'C0']})
        df2 = DataFrame({'Foo': Categorical(['C', 'B', 'A'], categories=['C', 'B', 'A']), 'Right': ['C1', 'B1', 'A1']})
        result = merge(df1, df2, on=['Foo'])
        expected = DataFrame({'Foo': Categorical(['A', 'B', 'C']), 'Left': ['A0', 'B0', 'C0'], 'Right': ['A1', 'B1', 'C1']})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [True, False])
    def test_multiindex_merge_with_unordered_categoricalindex(self, ordered):
        pcat = CategoricalDtype(categories=['P2', 'P1'], ordered=ordered)
        df1 = DataFrame({'id': ['C', 'C', 'D'], 'p': Categorical(['P2', 'P1', 'P2'], dtype=pcat), 'a': [0, 1, 2]}).set_index(['id', 'p'])
        df2 = DataFrame({'id': ['A', 'C', 'C'], 'p': Categorical(['P2', 'P2', 'P1'], dtype=pcat), 'd1': [10, 11, 12]}).set_index(['id', 'p'])
        result = merge(df1, df2, how='left', left_index=True, right_index=True)
        expected = DataFrame({'id': ['C', 'C', 'D'], 'p': Categorical(['P2', 'P1', 'P2'], dtype=pcat), 'a': [0, 1, 2], 'd1': [11.0, 12.0, np.nan]}).set_index(['id', 'p'])
        tm.assert_frame_equal(result, expected)

    def test_other_columns(self, left, right, using_infer_string):
        right = right.assign(Z=right.Z.astype('category'))
        merged = merge(left, right, on='X')
        result = merged.dtypes.sort_index()
        dtype = np.dtype('O') if not using_infer_string else 'string'
        expected = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, CategoricalDtype(categories=[1, 2])], index=['X', 'Y', 'Z'])
        tm.assert_series_equal(result, expected)
        assert left.X.values._categories_match_up_to_permutation(merged.X.values)
        assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)

    @pytest.mark.parametrize('change', [lambda x: x, lambda x: x.astype(CategoricalDtype(['foo', 'bar', 'bah'])), lambda x: x.astype(CategoricalDtype(ordered=True))])
    def test_dtype_on_merged_different(self, change, join_type, left, right, using_infer_string):
        X = change(right.X.astype('object'))
        right = right.assign(X=X)
        assert isinstance(left.X.values.dtype, CategoricalDtype)
        merged = merge(left, right, on='X', how=join_type)
        result = merged.dtypes.sort_index()
        dtype = np.dtype('O') if not using_infer_string else 'string'
        expected = Series([dtype, dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
        tm.assert_series_equal(result, expected)

    def test_self_join_multiple_categories(self):
        m = 5
        df = DataFrame({'a': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] * m, 'b': ['t', 'w', 'x', 'y', 'z'] * 2 * m, 'c': [letter for each in ['m', 'n', 'u', 'p', 'o'] for letter in [each] * 2 * m], 'd': [letter for each in ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'] for letter in [each] * m]})
        df = df.apply(lambda x: x.astype('category'))
        result = merge(df, df, on=list(df.columns))
        tm.assert_frame_equal(result, df)

    def test_dtype_on_categorical_dates(self):
        df = DataFrame([[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=['date', 'num2'])
        df['date'] = df['date'].astype('category')
        df2 = DataFrame([[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=['date', 'num4'])
        df2['date'] = df2['date'].astype('category')
        expected_outer = DataFrame([[pd.Timestamp('2001-01-01').date(), 1.1, 1.3], [pd.Timestamp('2001-01-02').date(), 1.3, np.nan], [pd.Timestamp('2001-01-03').date(), np.nan, 1.4]], columns=['date', 'num2', 'num4'])
        result_outer = merge(df, df2, how='outer', on=['date'])
        tm.assert_frame_equal(result_outer, expected_outer)
        expected_inner = DataFrame([[pd.Timestamp('2001-01-01').date(), 1.1, 1.3]], columns=['date', 'num2', 'num4'])
        result_inner = merge(df, df2, how='inner', on=['date'])
        tm.assert_frame_equal(result_inner, expected_inner)

    @pytest.mark.parametrize('ordered', [True, False])
    @pytest.mark.parametrize('category_column,categories,expected_categories', [([False, True, True, False], [True, False], [True, False]), ([2, 1, 1, 2], [1, 2], [1, 2]), (['False', 'True', 'True', 'False'], ['True', 'False'], ['True', 'False'])])
    def test_merging_with_bool_or_int_cateorical_column(self, category_column, categories, expected_categories, ordered):
        df1 = DataFrame({'id': [1, 2, 3, 4], 'cat': category_column})
        df1['cat'] = df1['cat'].astype(CategoricalDtype(categories, ordered=ordered))
        df2 = DataFrame({'id': [2, 4], 'num': [1, 9]})
        result = df1.merge(df2)
        expected = DataFrame({'id': [2, 4], 'cat': expected_categories, 'num': [1, 9]})
        expected['cat'] = expected['cat'].astype(CategoricalDtype(categories, ordered=ordered))
        tm.assert_frame_equal(expected, result)

    def test_merge_on_int_array(self):
        df = DataFrame({'A': Series([1, 2, np.nan], dtype='Int64'), 'B': 1})
        result = merge(df, df, on='A')
        expected = DataFrame({'A': Series([1, 2, np.nan], dtype='Int64'), 'B_x': 1, 'B_y': 1})
        tm.assert_frame_equal(result, expected)