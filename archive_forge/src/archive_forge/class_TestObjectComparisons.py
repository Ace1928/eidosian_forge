import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
class TestObjectComparisons:

    def test_comparison_object_numeric_nas(self, comparison_op):
        ser = Series(np.random.default_rng(2).standard_normal(10), dtype=object)
        shifted = ser.shift(2)
        func = comparison_op
        result = func(ser, shifted)
        expected = func(ser.astype(float), shifted.astype(float))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    def test_object_comparisons(self, infer_string):
        with option_context('future.infer_string', infer_string):
            ser = Series(['a', 'b', np.nan, 'c', 'a'])
            result = ser == 'a'
            expected = Series([True, False, False, False, True])
            tm.assert_series_equal(result, expected)
            result = ser < 'a'
            expected = Series([False, False, False, False, False])
            tm.assert_series_equal(result, expected)
            result = ser != 'a'
            expected = -(ser == 'a')
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', [None, object])
    def test_more_na_comparisons(self, dtype):
        left = Series(['a', np.nan, 'c'], dtype=dtype)
        right = Series(['a', np.nan, 'd'], dtype=dtype)
        result = left == right
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)
        result = left != right
        expected = Series([False, True, True])
        tm.assert_series_equal(result, expected)
        result = left == np.nan
        expected = Series([False, False, False])
        tm.assert_series_equal(result, expected)
        result = left != np.nan
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)