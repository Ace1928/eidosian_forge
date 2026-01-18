from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
def test_split_out_multi_column_groupby():
    df = pd.DataFrame({'x': np.arange(100) % 10, 'y': np.ones(100), 'z': [1, 2, 3, 4, 5] * 20})
    ddf = dd.from_pandas(df, npartitions=10)
    result = ddf.groupby(['x', 'y'], sort=False).z.mean(split_out=4)
    expected = df.groupby(['x', 'y']).z.mean()
    assert_eq(result, expected, check_dtype=False)