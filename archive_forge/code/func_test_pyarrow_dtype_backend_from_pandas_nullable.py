import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_pyarrow_dtype_backend_from_pandas_nullable(self):
    pa = pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': pd.Series([1, 2, None], dtype='Int32'), 'b': pd.Series(['x', 'y', None], dtype='string[python]'), 'c': pd.Series([True, False, None], dtype='boolean'), 'd': pd.Series([None, 100.5, 200], dtype='Float64')})
    result = df.convert_dtypes(dtype_backend='pyarrow')
    expected = pd.DataFrame({'a': pd.arrays.ArrowExtensionArray(pa.array([1, 2, None], type=pa.int32())), 'b': pd.arrays.ArrowExtensionArray(pa.array(['x', 'y', None])), 'c': pd.arrays.ArrowExtensionArray(pa.array([True, False, None])), 'd': pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0]))})
    tm.assert_frame_equal(result, expected)