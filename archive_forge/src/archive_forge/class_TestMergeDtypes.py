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
class TestMergeDtypes:

    @pytest.mark.parametrize('right_vals', [['foo', 'bar'], Series(['foo', 'bar']).astype('category')])
    def test_different(self, right_vals):
        left = DataFrame({'A': ['foo', 'bar'], 'B': Series(['foo', 'bar']).astype('category'), 'C': [1, 2], 'D': [1.0, 2.0], 'E': Series([1, 2], dtype='uint64'), 'F': Series([1, 2], dtype='int32')})
        right = DataFrame({'A': right_vals})
        result = merge(left, right, on='A')
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize('d1', [np.int64, np.int32, np.intc, np.int16, np.int8, np.uint8])
    @pytest.mark.parametrize('d2', [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(self, d1, d2):
        dtype1 = np.dtype(d1)
        dtype2 = np.dtype(d2)
        left = DataFrame({'k1': np.array([0, 1, 2] * 8, dtype=dtype1), 'k2': ['foo', 'bar'] * 12, 'v': np.array(np.arange(24), dtype=np.int64)})
        index = MultiIndex.from_tuples([(2, 'bar'), (1, 'foo')])
        right = DataFrame({'v2': np.array([5, 7], dtype=dtype2)}, index=index)
        result = left.join(right, on=['k1', 'k2'])
        expected = left.copy()
        if dtype2.kind == 'i':
            dtype2 = np.dtype('float64')
        expected['v2'] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == 'bar'), 'v2'] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == 'foo'), 'v2'] = 7
        tm.assert_frame_equal(result, expected)
        result = left.join(right, on=['k1', 'k2'], sort=True)
        expected.sort_values(['k1', 'k2'], kind='mergesort', inplace=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('int_vals, float_vals, exp_vals', [([1, 2, 3], [1.0, 2.0, 3.0], {'X': [1, 2, 3], 'Y': [1.0, 2.0, 3.0]}), ([1, 2, 3], [1.0, 3.0], {'X': [1, 3], 'Y': [1.0, 3.0]}), ([1, 2], [1.0, 2.0, 3.0], {'X': [1, 2], 'Y': [1.0, 2.0]})])
    def test_merge_on_ints_floats(self, int_vals, float_vals, exp_vals):
        A = DataFrame({'X': int_vals})
        B = DataFrame({'Y': float_vals})
        expected = DataFrame(exp_vals)
        result = A.merge(B, left_on='X', right_on='Y')
        tm.assert_frame_equal(result, expected)
        result = B.merge(A, left_on='Y', right_on='X')
        tm.assert_frame_equal(result, expected[['Y', 'X']])

    def test_merge_key_dtype_cast(self):
        df1 = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20]}, columns=['key', 'v1'])
        df2 = DataFrame({'key': [2], 'v2': [200]}, columns=['key', 'v2'])
        result = df1.merge(df2, on='key', how='left')
        expected = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20], 'v2': [np.nan, 200.0]}, columns=['key', 'v1', 'v2'])
        tm.assert_frame_equal(result, expected)

    def test_merge_on_ints_floats_warning(self):
        A = DataFrame({'X': [1, 2, 3]})
        B = DataFrame({'Y': [1.1, 2.5, 3.0]})
        expected = DataFrame({'X': [3], 'Y': [3.0]})
        with tm.assert_produces_warning(UserWarning):
            result = A.merge(B, left_on='X', right_on='Y')
            tm.assert_frame_equal(result, expected)
        with tm.assert_produces_warning(UserWarning):
            result = B.merge(A, left_on='Y', right_on='X')
            tm.assert_frame_equal(result, expected[['Y', 'X']])
        B = DataFrame({'Y': [np.nan, np.nan, 3.0]})
        with tm.assert_produces_warning(None):
            result = B.merge(A, left_on='Y', right_on='X')
            tm.assert_frame_equal(result, expected[['Y', 'X']])

    def test_merge_incompat_infer_boolean_object(self):
        df1 = DataFrame({'key': Series([True, False], dtype=object)})
        df2 = DataFrame({'key': [True, False]})
        expected = DataFrame({'key': [True, False]}, dtype=object)
        result = merge(df1, df2, on='key')
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on='key')
        tm.assert_frame_equal(result, expected)

    def test_merge_incompat_infer_boolean_object_with_missing(self):
        df1 = DataFrame({'key': Series([True, False, np.nan], dtype=object)})
        df2 = DataFrame({'key': [True, False]})
        expected = DataFrame({'key': [True, False]}, dtype=object)
        result = merge(df1, df2, on='key')
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on='key')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('df1_vals, df2_vals', [([0, 1, 2], Series(['a', 'b', 'a']).astype('category')), ([0.0, 1.0, 2.0], Series(['a', 'b', 'a']).astype('category')), ([0, 1], Series([False, True], dtype=object)), ([0, 1], Series([False, True], dtype=bool))])
    def test_merge_incompat_dtypes_are_ok(self, df1_vals, df2_vals):
        df1 = DataFrame({'A': df1_vals})
        df2 = DataFrame({'A': df2_vals})
        result = merge(df1, df2, on=['A'])
        assert is_object_dtype(result.A.dtype)
        result = merge(df2, df1, on=['A'])
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize('df1_vals, df2_vals', [(Series([1, 2], dtype='uint64'), ['a', 'b', 'c']), (Series([1, 2], dtype='int32'), ['a', 'b', 'c']), ([0, 1, 2], ['0', '1', '2']), ([0.0, 1.0, 2.0], ['0', '1', '2']), ([0, 1, 2], ['0', '1', '2']), (pd.date_range('1/1/2011', periods=2, freq='D'), ['2011-01-01', '2011-01-02']), (pd.date_range('1/1/2011', periods=2, freq='D'), [0, 1]), (pd.date_range('1/1/2011', periods=2, freq='D'), [0.0, 1.0]), (pd.date_range('20130101', periods=3), pd.date_range('20130101', periods=3, tz='US/Eastern'))])
    def test_merge_incompat_dtypes_error(self, df1_vals, df2_vals):
        df1 = DataFrame({'A': df1_vals})
        df2 = DataFrame({'A': df2_vals})
        msg = f"You are trying to merge on {df1['A'].dtype} and {df2['A'].dtype} columns for key 'A'. If you wish to proceed you should use pd.concat"
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on=['A'])
        msg = f"You are trying to merge on {df2['A'].dtype} and {df1['A'].dtype} columns for key 'A'. If you wish to proceed you should use pd.concat"
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df2, df1, on=['A'])
        if len(df1_vals) == len(df2_vals):
            df3 = DataFrame({'A': df2_vals, 'B': df1_vals, 'C': df1_vals})
            df4 = DataFrame({'A': df2_vals, 'B': df2_vals, 'C': df2_vals})
            msg = f"You are trying to merge on {df3['B'].dtype} and {df4['B'].dtype} columns for key 'B'. If you wish to proceed you should use pd.concat"
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4)
            msg = f"You are trying to merge on {df3['C'].dtype} and {df4['C'].dtype} columns for key 'C'. If you wish to proceed you should use pd.concat"
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4, on=['A', 'C'])

    @pytest.mark.parametrize('expected_data, how', [([1, 2], 'outer'), ([], 'inner'), ([2], 'right'), ([1], 'left')])
    def test_merge_EA_dtype(self, any_numeric_ea_dtype, how, expected_data):
        d1 = DataFrame([(1,)], columns=['id'], dtype=any_numeric_ea_dtype)
        d2 = DataFrame([(2,)], columns=['id'], dtype=any_numeric_ea_dtype)
        result = merge(d1, d2, how=how)
        exp_index = RangeIndex(len(expected_data))
        expected = DataFrame(expected_data, index=exp_index, columns=['id'], dtype=any_numeric_ea_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('expected_data, how', [(['a', 'b'], 'outer'), ([], 'inner'), (['b'], 'right'), (['a'], 'left')])
    def test_merge_string_dtype(self, how, expected_data, any_string_dtype):
        d1 = DataFrame([('a',)], columns=['id'], dtype=any_string_dtype)
        d2 = DataFrame([('b',)], columns=['id'], dtype=any_string_dtype)
        result = merge(d1, d2, how=how)
        exp_idx = RangeIndex(len(expected_data))
        expected = DataFrame(expected_data, index=exp_idx, columns=['id'], dtype=any_string_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how, expected_data', [('inner', [[True, 1, 4], [False, 5, 3]]), ('outer', [[False, 5, 3], [True, 1, 4]]), ('left', [[True, 1, 4], [False, 5, 3]]), ('right', [[False, 5, 3], [True, 1, 4]])])
    def test_merge_bool_dtype(self, how, expected_data):
        df1 = DataFrame({'A': [True, False], 'B': [1, 5]})
        df2 = DataFrame({'A': [False, True], 'C': [3, 4]})
        result = merge(df1, df2, how=how)
        expected = DataFrame(expected_data, columns=['A', 'B', 'C'])
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_with_string(self, join_type, string_dtype):
        df1 = DataFrame(data={('lvl0', 'lvl1-a'): ['1', '2', '3', '4', None], ('lvl0', 'lvl1-b'): ['4', '5', '6', '7', '8']}, dtype=pd.StringDtype())
        df1_copy = df1.copy()
        df2 = DataFrame(data={('lvl0', 'lvl1-a'): ['1', '2', '3', pd.NA, '5'], ('lvl0', 'lvl1-c'): ['7', '8', '9', pd.NA, '11']}, dtype=string_dtype)
        df2_copy = df2.copy()
        merged = merge(left=df1, right=df2, on=[('lvl0', 'lvl1-a')], how=join_type)
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)
        expected = Series([np.dtype('O'), pd.StringDtype(), np.dtype('O')], index=MultiIndex.from_tuples([('lvl0', 'lvl1-a'), ('lvl0', 'lvl1-b'), ('lvl0', 'lvl1-c')]))
        tm.assert_series_equal(merged.dtypes, expected)

    @pytest.mark.parametrize('left_empty, how, exp', [(False, 'left', 'left'), (False, 'right', 'empty'), (False, 'inner', 'empty'), (False, 'outer', 'left'), (False, 'cross', 'empty_cross'), (True, 'left', 'empty'), (True, 'right', 'right'), (True, 'inner', 'empty'), (True, 'outer', 'right'), (True, 'cross', 'empty_cross')])
    def test_merge_empty(self, left_empty, how, exp):
        left = DataFrame({'A': [2, 1], 'B': [3, 4]})
        right = DataFrame({'A': [1], 'C': [5]}, dtype='int64')
        if left_empty:
            left = left.head(0)
        else:
            right = right.head(0)
        result = left.merge(right, how=how)
        if exp == 'left':
            expected = DataFrame({'A': [2, 1], 'B': [3, 4], 'C': [np.nan, np.nan]})
        elif exp == 'right':
            expected = DataFrame({'A': [1], 'B': [np.nan], 'C': [5]})
        elif exp == 'empty':
            expected = DataFrame(columns=['A', 'B', 'C'], dtype='int64')
        elif exp == 'empty_cross':
            expected = DataFrame(columns=['A_x', 'B', 'A_y', 'C'], dtype='int64')
        if how == 'outer':
            expected = expected.sort_values('A', ignore_index=True)
        tm.assert_frame_equal(result, expected)