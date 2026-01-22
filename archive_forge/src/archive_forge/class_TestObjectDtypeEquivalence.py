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
class TestObjectDtypeEquivalence:

    @pytest.mark.parametrize('dtype', [None, object])
    def test_numarr_with_dtype_add_nan(self, dtype, box_with_array):
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([np.nan, np.nan, np.nan], dtype=dtype)
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)
        result = np.nan + ser
        tm.assert_equal(result, expected)
        result = ser + np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('dtype', [None, object])
    def test_numarr_with_dtype_add_int(self, dtype, box_with_array):
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([2, 3, 4], dtype=dtype)
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)
        result = 1 + ser
        tm.assert_equal(result, expected)
        result = ser + 1
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv])
    def test_operators_reverse_object(self, op):
        arr = Series(np.random.default_rng(2).standard_normal(10), index=np.arange(10), dtype=object)
        result = op(1.0, arr)
        expected = op(1.0, arr.astype(float))
        tm.assert_series_equal(result.astype(float), expected)