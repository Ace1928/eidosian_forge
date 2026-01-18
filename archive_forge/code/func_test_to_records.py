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
def test_to_records():
    pytest.importorskip('dask.array')
    from dask.array.utils import assert_eq
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [2, 3, 4, 5]}, index=pd.Index([1.0, 2.0, 3.0, 4.0], name='ind'))
    ddf = dd.from_pandas(df, 2)
    assert_eq(df.to_records(), ddf.to_records(), check_type=False)