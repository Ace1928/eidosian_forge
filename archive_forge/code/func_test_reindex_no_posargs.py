import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_no_posargs():
    ser = Series([1, 2])
    result = ser.reindex(index=[1, 0])
    expected = Series([2, 1], index=[1, 0])
    tm.assert_series_equal(result, expected)