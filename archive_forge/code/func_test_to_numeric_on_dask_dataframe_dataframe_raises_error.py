from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from dask.array import Array, from_array
from dask.dataframe import Series, _dask_expr_enabled, from_pandas, to_numeric
from dask.dataframe.utils import pyarrow_strings_enabled
from dask.delayed import Delayed
def test_to_numeric_on_dask_dataframe_dataframe_raises_error():
    s = pd.Series(['1.0', '2', -3, -5.1])
    df = pd.DataFrame({'a': s, 'b': s})
    arg = from_pandas(df, npartitions=2)
    with pytest.raises(TypeError, match='arg must be a list, tuple, dask.'):
        to_numeric(arg)