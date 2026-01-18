import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean')])
def test_to_numeric_dtype_backend(val, dtype):
    ser = Series([val], dtype=object)
    result = to_numeric(ser, dtype_backend='numpy_nullable')
    expected = Series([val], dtype=dtype)
    tm.assert_series_equal(result, expected)