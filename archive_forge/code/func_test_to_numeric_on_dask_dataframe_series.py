from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from dask.array import Array, from_array
from dask.dataframe import Series, _dask_expr_enabled, from_pandas, to_numeric
from dask.dataframe.utils import pyarrow_strings_enabled
from dask.delayed import Delayed
def test_to_numeric_on_dask_dataframe_series():
    s = pd.Series(['1.0', '2', -3, -5.1])
    arg = from_pandas(s, npartitions=2)
    expected = pd.to_numeric(s)
    output = to_numeric(arg)
    expected_dtype = 'int64'
    if pyarrow_strings_enabled():
        expected_dtype = 'Int64'
    assert output.dtype == expected_dtype
    assert isinstance(output, Series)
    assert list(output.compute()) == list(expected)