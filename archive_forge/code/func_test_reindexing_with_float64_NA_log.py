import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindexing_with_float64_NA_log():
    s = Series([1.0, NA], dtype=Float64Dtype())
    s_reindex = s.reindex(range(3))
    result = s_reindex.values._data
    expected = np.array([1, np.nan, np.nan])
    tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(None):
        result_log = np.log(s_reindex)
        expected_log = Series([0, np.nan, np.nan], dtype=Float64Dtype())
        tm.assert_series_equal(result_log, expected_log)