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
@pytest.mark.parametrize('dtype1', [np.int64, np.float64, np.uint64])
def test_ser_divmod_zero(self, dtype1, any_real_numpy_dtype):
    dtype2 = any_real_numpy_dtype
    left = Series([1, 1]).astype(dtype1)
    right = Series([0, 2]).astype(dtype2)
    expected = (left // right, left % right)
    expected = list(expected)
    expected[0] = expected[0].astype(np.float64)
    expected[0][0] = np.inf
    result = divmod(left, right)
    tm.assert_series_equal(result[0], expected[0])
    tm.assert_series_equal(result[1], expected[1])
    result = divmod(left.values, right)
    tm.assert_series_equal(result[0], expected[0])
    tm.assert_series_equal(result[1], expected[1])