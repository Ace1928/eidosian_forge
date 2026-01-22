from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
class TestFrameFlexArithmetic:

    def test_floordiv_axis0(self):
        arr = np.arange(3)
        ser = Series(arr)
        df = DataFrame({'A': ser, 'B': ser})
        result = df.floordiv(ser, axis=0)
        expected = DataFrame({col: df[col] // ser for col in df.columns})
        tm.assert_frame_equal(result, expected)
        result2 = df.floordiv(ser.values, axis=0)
        tm.assert_frame_equal(result2, expected)

    def test_df_add_td64_columnwise(self):
        dti = pd.date_range('2016-01-01', periods=10)
        tdi = pd.timedelta_range('1', periods=10)
        tser = Series(tdi)
        df = DataFrame({0: dti, 1: tdi})
        result = df.add(tser, axis=0)
        expected = DataFrame({0: dti + tdi, 1: tdi + tdi})
        tm.assert_frame_equal(result, expected)

    def test_df_add_flex_filled_mixed_dtypes(self):
        dti = pd.date_range('2016-01-01', periods=3)
        ser = Series(['1 Day', 'NaT', '2 Days'], dtype='timedelta64[ns]')
        df = DataFrame({'A': dti, 'B': ser})
        other = DataFrame({'A': ser, 'B': ser})
        fill = pd.Timedelta(days=1).to_timedelta64()
        result = df.add(other, fill_value=fill)
        expected = DataFrame({'A': Series(['2016-01-02', '2016-01-03', '2016-01-05'], dtype='datetime64[ns]'), 'B': ser * 2})
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_frame(self, all_arithmetic_operators, float_frame, mixed_float_frame):
        op = all_arithmetic_operators

        def f(x, y):
            if op.startswith('__r'):
                return getattr(operator, op.replace('__r', '__'))(y, x)
            return getattr(operator, op)(x, y)
        result = getattr(float_frame, op)(2 * float_frame)
        expected = f(float_frame, 2 * float_frame)
        tm.assert_frame_equal(result, expected)
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={'C': None})

    @pytest.mark.parametrize('op', ['__add__', '__sub__', '__mul__'])
    def test_arith_flex_frame_mixed(self, op, int_frame, mixed_int_frame, mixed_float_frame, switch_numexpr_min_elements):
        f = getattr(operator, op)
        result = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
        expected = f(mixed_int_frame, 2 + mixed_int_frame)
        dtype = None
        if op in ['__sub__']:
            dtype = {'B': 'uint64', 'C': None}
        elif op in ['__add__', '__mul__']:
            dtype = {'C': None}
        if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
            dtype['A'] = (2 + mixed_int_frame)['A'].dtype
        tm.assert_frame_equal(result, expected)
        _check_mixed_int(result, dtype=dtype)
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={'C': None})
        result = getattr(int_frame, op)(2 * int_frame)
        expected = f(int_frame, 2 * int_frame)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dim', range(3, 6))
    def test_arith_flex_frame_raise(self, all_arithmetic_operators, float_frame, dim):
        op = all_arithmetic_operators
        arr = np.ones((1,) * dim)
        msg = 'Unable to coerce to Series/DataFrame'
        with pytest.raises(ValueError, match=msg):
            getattr(float_frame, op)(arr)

    def test_arith_flex_frame_corner(self, float_frame):
        const_add = float_frame.add(1)
        tm.assert_frame_equal(const_add, float_frame + 1)
        result = float_frame.add(float_frame[:0])
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)
        result = float_frame[:0].add(float_frame)
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)
        with pytest.raises(NotImplementedError, match='fill_value'):
            float_frame.add(float_frame.iloc[0], fill_value=3)
        with pytest.raises(NotImplementedError, match='fill_value'):
            float_frame.add(float_frame.iloc[0], axis='index', fill_value=3)

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'mod'])
    def test_arith_flex_series_ops(self, simple_frame, op):
        df = simple_frame
        row = df.xs('a')
        col = df['two']
        f = getattr(df, op)
        op = getattr(operator, op)
        tm.assert_frame_equal(f(row), op(df, row))
        tm.assert_frame_equal(f(col, axis=0), op(df.T, col).T)

    def test_arith_flex_series(self, simple_frame):
        df = simple_frame
        row = df.xs('a')
        col = df['two']
        tm.assert_frame_equal(df.add(row, axis=None), df + row)
        tm.assert_frame_equal(df.div(row), df / row)
        tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)

    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    def test_arith_flex_series_broadcasting(self, dtype):
        df = DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=dtype)
        expected = DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]])
        result = df.div(df[0], axis='index')
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_zero_len_raises(self):
        ser_len0 = Series([], dtype=object)
        df_len0 = DataFrame(columns=['A', 'B'])
        df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        with pytest.raises(NotImplementedError, match='fill_value'):
            df.add(ser_len0, fill_value='E')
        with pytest.raises(NotImplementedError, match='fill_value'):
            df_len0.sub(df['A'], axis=None, fill_value=3)

    def test_flex_add_scalar_fill_value(self):
        dat = np.array([0, 1, np.nan, 3, 4, 5], dtype='float')
        df = DataFrame({'foo': dat}, index=range(6))
        exp = df.fillna(0).add(2)
        res = df.add(2, fill_value=0)
        tm.assert_frame_equal(res, exp)

    def test_sub_alignment_with_duplicate_index(self):
        df1 = DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
        df2 = DataFrame([1, 2, 3], index=[1, 2, 3])
        expected = DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
        result = df1.sub(df2)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op', ['__add__', '__mul__', '__sub__', '__truediv__'])
    def test_arithmetic_with_duplicate_columns(self, op):
        df = DataFrame({'A': np.arange(10), 'B': np.random.default_rng(2).random(10)})
        expected = getattr(df, op)(df)
        expected.columns = ['A', 'A']
        df.columns = ['A', 'A']
        result = getattr(df, op)(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('level', [0, None])
    def test_broadcast_multiindex(self, level):
        df1 = DataFrame({'A': [0, 1, 2], 'B': [1, 2, 3]})
        df1.columns = df1.columns.set_names('L1')
        df2 = DataFrame({('A', 'C'): [0, 0, 0], ('A', 'D'): [0, 0, 0]})
        df2.columns = df2.columns.set_names(['L1', 'L2'])
        result = df1.add(df2, level=level)
        expected = DataFrame({('A', 'C'): [0, 1, 2], ('A', 'D'): [0, 1, 2]})
        expected.columns = expected.columns.set_names(['L1', 'L2'])
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations(self):
        df = DataFrame({2010: [1, 2, 3], 2020: [3, 4, 5]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        series = Series([0.4], index=MultiIndex.from_product([['b'], ['a']], names=['mod', 'scen']))
        expected = DataFrame({2010: [1.4, 2.4, 3.4], 2020: [3.4, 4.4, 5.4]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        result = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_series_index_to_frame_index(self):
        df = DataFrame({2010: [1], 2020: [3]}, index=MultiIndex.from_product([['a'], ['b']], names=['scen', 'mod']))
        series = Series([10.0, 20.0, 30.0], index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        expected = DataFrame({2010: [11.0, 21, 31.0], 2020: [13.0, 23.0, 33.0]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        result = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_no_align(self):
        df = DataFrame({2010: [1, 2, 3], 2020: [3, 4, 5]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        series = Series([0.4], index=MultiIndex.from_product([['c'], ['a']], names=['mod', 'scen']))
        expected = DataFrame({2010: np.nan, 2020: np.nan}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'b', 2), ('a', 'c', np.nan)], names=['scen', 'mod', 'id']))
        result = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_part_align(self):
        df = DataFrame({2010: [1, 2, 3], 2020: [3, 4, 5]}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'c', 2)], names=['scen', 'mod', 'id']))
        series = Series([0.4], index=MultiIndex.from_product([['b'], ['a']], names=['mod', 'scen']))
        expected = DataFrame({2010: [1.4, 2.4, np.nan], 2020: [3.4, 4.4, np.nan]}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'c', 2)], names=['scen', 'mod', 'id']))
        result = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)