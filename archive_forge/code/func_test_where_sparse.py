import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_sparse():
    ser = Series(pd.arrays.SparseArray([1, 2]))
    result = ser.where(ser >= 2, 0)
    expected = Series(pd.arrays.SparseArray([0, 2]))
    tm.assert_series_equal(result, expected)