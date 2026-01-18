import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray
import pyarrow as pa
def test_orc_dtype_backend_numpy_nullable():
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'string': list('abc'), 'string_with_nan': ['a', np.nan, 'c'], 'string_with_none': ['a', None, 'c'], 'int': list(range(1, 4)), 'int_with_nan': pd.Series([1, pd.NA, 3], dtype='Int64'), 'na_only': pd.Series([pd.NA, pd.NA, pd.NA], dtype='Int64'), 'float': np.arange(4.0, 7.0, dtype='float64'), 'float_with_nan': [2.0, np.nan, 3.0], 'bool': [True, False, True], 'bool_with_na': [True, False, None]})
    bytes_data = df.copy().to_orc()
    result = read_orc(BytesIO(bytes_data), dtype_backend='numpy_nullable')
    expected = pd.DataFrame({'string': StringArray(np.array(['a', 'b', 'c'], dtype=np.object_)), 'string_with_nan': StringArray(np.array(['a', pd.NA, 'c'], dtype=np.object_)), 'string_with_none': StringArray(np.array(['a', pd.NA, 'c'], dtype=np.object_)), 'int': pd.Series([1, 2, 3], dtype='Int64'), 'int_with_nan': pd.Series([1, pd.NA, 3], dtype='Int64'), 'na_only': pd.Series([pd.NA, pd.NA, pd.NA], dtype='Int64'), 'float': pd.Series([4.0, 5.0, 6.0], dtype='Float64'), 'float_with_nan': pd.Series([2.0, pd.NA, 3.0], dtype='Float64'), 'bool': pd.Series([True, False, True], dtype='boolean'), 'bool_with_na': pd.Series([True, False, pd.NA], dtype='boolean')})
    tm.assert_frame_equal(result, expected)