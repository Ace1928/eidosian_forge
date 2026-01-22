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
class TestNumericComparisons:

    def test_operator_series_comparison_zerorank(self):
        result = np.float64(0) > Series([1, 2, 3])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)
        result = Series([1, 2, 3]) < np.float64(0)
        expected = Series([1, 2, 3]) < 0.0
        tm.assert_series_equal(result, expected)
        result = np.array([0, 1, 2])[0] > Series([0, 1, 2])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_df_numeric_cmp_dt64_raises(self, box_with_array, fixed_now_ts):
        ts = fixed_now_ts
        obj = np.array(range(5))
        obj = tm.box_expected(obj, box_with_array)
        assert_invalid_comparison(obj, ts, box_with_array)

    def test_compare_invalid(self):
        a = Series(np.random.default_rng(2).standard_normal(5), name=0)
        b = Series(np.random.default_rng(2).standard_normal(5))
        b.name = pd.Timestamp('2000-01-01')
        tm.assert_series_equal(a / b, 1 / (b / a))

    def test_numeric_cmp_string_numexpr_path(self, box_with_array, monkeypatch):
        box = box_with_array
        xbox = box if box is not Index else np.ndarray
        obj = Series(np.random.default_rng(2).standard_normal(51))
        obj = tm.box_expected(obj, box, transpose=False)
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 50)
            result = obj == 'a'
        expected = Series(np.zeros(51, dtype=bool))
        expected = tm.box_expected(expected, xbox, transpose=False)
        tm.assert_equal(result, expected)
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 50)
            result = obj != 'a'
        tm.assert_equal(result, ~expected)
        msg = 'Invalid comparison between dtype=float64 and str'
        with pytest.raises(TypeError, match=msg):
            obj < 'a'