from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
class TestNumericArraylikeArithmeticWithDatetimeLike:

    @pytest.mark.parametrize('box_cls', [np.array, Index, Series])
    @pytest.mark.parametrize('left', lefts, ids=lambda x: type(x).__name__ + str(x.dtype))
    def test_mul_td64arr(self, left, box_cls):
        right = np.array([1, 2, 3], dtype='m8[s]')
        right = box_cls(right)
        expected = TimedeltaIndex(['10s', '40s', '90s'], dtype=right.dtype)
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        assert expected.dtype == right.dtype
        result = left * right
        tm.assert_equal(result, expected)
        result = right * left
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('box_cls', [np.array, Index, Series])
    @pytest.mark.parametrize('left', lefts, ids=lambda x: type(x).__name__ + str(x.dtype))
    def test_div_td64arr(self, left, box_cls):
        right = np.array([10, 40, 90], dtype='m8[s]')
        right = box_cls(right)
        expected = TimedeltaIndex(['1s', '2s', '3s'], dtype=right.dtype)
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        assert expected.dtype == right.dtype
        result = right / left
        tm.assert_equal(result, expected)
        result = right // left
        tm.assert_equal(result, expected)
        msg = "ufunc '(true_)?divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left / right
        msg = "ufunc 'floor_divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left // right

    @pytest.mark.parametrize('scalar_td', [Timedelta(days=1), Timedelta(days=1).to_timedelta64(), Timedelta(days=1).to_pytimedelta(), Timedelta(days=1).to_timedelta64().astype('timedelta64[s]'), Timedelta(days=1).to_timedelta64().astype('timedelta64[ms]')], ids=lambda x: type(x).__name__)
    def test_numeric_arr_mul_tdscalar(self, scalar_td, numeric_idx, box_with_array):
        box = box_with_array
        index = numeric_idx
        expected = TimedeltaIndex([Timedelta(days=n) for n in range(len(index))])
        if isinstance(scalar_td, np.timedelta64):
            dtype = scalar_td.dtype
            expected = expected.astype(dtype)
        elif type(scalar_td) is timedelta:
            expected = expected.astype('m8[us]')
        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)
        result = index * scalar_td
        tm.assert_equal(result, expected)
        commute = scalar_td * index
        tm.assert_equal(commute, expected)

    @pytest.mark.parametrize('scalar_td', [Timedelta(days=1), Timedelta(days=1).to_timedelta64(), Timedelta(days=1).to_pytimedelta()], ids=lambda x: type(x).__name__)
    @pytest.mark.parametrize('dtype', [np.int64, np.float64])
    def test_numeric_arr_mul_tdscalar_numexpr_path(self, dtype, scalar_td, box_with_array):
        box = box_with_array
        arr_i8 = np.arange(2 * 10 ** 4).astype(np.int64, copy=False)
        arr = arr_i8.astype(dtype, copy=False)
        obj = tm.box_expected(arr, box, transpose=False)
        expected = arr_i8.view('timedelta64[D]').astype('timedelta64[ns]')
        if type(scalar_td) is timedelta:
            expected = expected.astype('timedelta64[us]')
        expected = tm.box_expected(expected, box, transpose=False)
        result = obj * scalar_td
        tm.assert_equal(result, expected)
        result = scalar_td * obj
        tm.assert_equal(result, expected)

    def test_numeric_arr_rdiv_tdscalar(self, three_days, numeric_idx, box_with_array):
        box = box_with_array
        index = numeric_idx[1:3]
        expected = TimedeltaIndex(['3 Days', '36 Hours'])
        if isinstance(three_days, np.timedelta64):
            dtype = three_days.dtype
            if dtype < np.dtype('m8[s]'):
                dtype = np.dtype('m8[s]')
            expected = expected.astype(dtype)
        elif type(three_days) is timedelta:
            expected = expected.astype('m8[us]')
        elif isinstance(three_days, (pd.offsets.Day, pd.offsets.Hour, pd.offsets.Minute, pd.offsets.Second)):
            expected = expected.astype('m8[s]')
        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)
        result = three_days / index
        tm.assert_equal(result, expected)
        msg = 'cannot use operands with types dtype'
        with pytest.raises(TypeError, match=msg):
            index / three_days

    @pytest.mark.parametrize('other', [Timedelta(hours=31), Timedelta(hours=31).to_pytimedelta(), Timedelta(hours=31).to_timedelta64(), Timedelta(hours=31).to_timedelta64().astype('m8[h]'), np.timedelta64('NaT'), np.timedelta64('NaT', 'D'), pd.offsets.Minute(3), pd.offsets.Second(0), pd.Timestamp('2021-01-01', tz='Asia/Tokyo'), pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-01').to_pydatetime(), pd.Timestamp('2021-01-01', tz='UTC').to_pydatetime(), pd.Timestamp('2021-01-01').to_datetime64(), np.datetime64('NaT', 'ns'), pd.NaT], ids=repr)
    def test_add_sub_datetimedeltalike_invalid(self, numeric_idx, other, box_with_array):
        box = box_with_array
        left = tm.box_expected(numeric_idx, box)
        msg = '|'.join(['unsupported operand type', 'Addition/subtraction of integers and integer-arrays', 'Instead of adding/subtracting', 'cannot use operands with types dtype', 'Concatenation operation is not implemented for NumPy arrays', 'Cannot (add|subtract) NaT (to|from) ndarray', 'operand type\\(s\\) all returned NotImplemented from __array_ufunc__', 'can only perform ops with numeric values', 'cannot subtract DatetimeArray from ndarray', 'Cannot add or subtract Timedelta from integers'])
        assert_invalid_addsub_type(left, other, msg)