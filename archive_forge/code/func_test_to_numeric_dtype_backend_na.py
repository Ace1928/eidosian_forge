import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean'), (1, 'int64[pyarrow]'), (1.5, 'float64[pyarrow]'), (True, 'bool[pyarrow]')])
def test_to_numeric_dtype_backend_na(val, dtype):
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
        dtype_backend = 'pyarrow'
    else:
        dtype_backend = 'numpy_nullable'
    ser = Series([val, None], dtype=object)
    result = to_numeric(ser, dtype_backend=dtype_backend)
    expected = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)