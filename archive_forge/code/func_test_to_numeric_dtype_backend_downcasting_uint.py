import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('smaller, dtype_backend', [['UInt8', 'numpy_nullable'], ['uint8[pyarrow]', 'pyarrow']])
def test_to_numeric_dtype_backend_downcasting_uint(smaller, dtype_backend):
    if dtype_backend == 'pyarrow':
        pytest.importorskip('pyarrow')
    ser = Series([1, pd.NA], dtype='UInt64')
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast='unsigned')
    expected = Series([1, pd.NA], dtype=smaller)
    tm.assert_series_equal(result, expected)