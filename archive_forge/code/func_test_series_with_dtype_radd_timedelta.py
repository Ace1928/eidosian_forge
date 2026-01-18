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
@pytest.mark.parametrize('dtype', [None, object])
def test_series_with_dtype_radd_timedelta(self, dtype):
    ser = Series([pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days')], dtype=dtype)
    expected = Series([pd.Timedelta('4 days'), pd.Timedelta('5 days'), pd.Timedelta('6 days')], dtype=dtype)
    result = pd.Timedelta('3 days') + ser
    tm.assert_series_equal(result, expected)
    result = ser + pd.Timedelta('3 days')
    tm.assert_series_equal(result, expected)