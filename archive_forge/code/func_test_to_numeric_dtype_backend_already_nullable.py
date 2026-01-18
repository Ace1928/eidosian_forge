import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int64', 'UInt64', 'Float64', 'boolean', 'int64[pyarrow]', 'uint64[pyarrow]', 'float64[pyarrow]', 'bool[pyarrow]'])
def test_to_numeric_dtype_backend_already_nullable(dtype):
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
    ser = Series([1, pd.NA], dtype=dtype)
    result = to_numeric(ser, dtype_backend='numpy_nullable')
    expected = Series([1, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)