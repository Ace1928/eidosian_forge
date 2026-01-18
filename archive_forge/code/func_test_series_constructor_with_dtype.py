from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_series_constructor_with_dtype():
    arr = DecimalArray([decimal.Decimal('10.0')])
    result = pd.Series(arr, dtype=DecimalDtype())
    expected = pd.Series(arr)
    tm.assert_series_equal(result, expected)
    result = pd.Series(arr, dtype='int64')
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)