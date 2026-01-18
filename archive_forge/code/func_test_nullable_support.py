import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
@pytest.mark.parametrize('dtype', [pd.BooleanDtype, pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype, pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype])
def test_nullable_support(dtype, version):
    df = DataFrame({'a': Series([1.0, 2.0, 3.0]), 'b': Series([1, pd.NA, pd.NA], dtype=dtype.name), 'c': Series(['a', 'b', None])})
    dtype_name = df.b.dtype.numpy_dtype.name
    dtype_name = dtype_name.replace('u', '')
    if dtype_name == 'int64':
        dtype_name = 'int32'
    elif dtype_name == 'bool':
        dtype_name = 'int8'
    value = StataMissingValue.BASE_MISSING_VALUES[dtype_name]
    smv = StataMissingValue(value)
    expected_b = Series([1, smv, smv], dtype=object, name='b')
    expected_c = Series(['a', 'b', ''], name='c')
    with tm.ensure_clean() as path:
        df.to_stata(path, write_index=False, version=version)
        reread = read_stata(path, convert_missing=True)
        tm.assert_series_equal(df.a, reread.a)
        tm.assert_series_equal(reread.b, expected_b)
        tm.assert_series_equal(reread.c, expected_c)