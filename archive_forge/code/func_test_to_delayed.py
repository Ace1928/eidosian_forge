from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
def test_to_delayed():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40]})
    ddf = dd.from_pandas(df, npartitions=2)
    a, b = ddf.to_delayed()
    assert isinstance(a, Delayed)
    assert isinstance(b, Delayed)
    assert_eq(a.compute(), df.iloc[:2])
    x = ddf.x.sum()
    dx = x.to_delayed()
    assert isinstance(dx, Delayed)
    assert_eq(dx.compute(), x)