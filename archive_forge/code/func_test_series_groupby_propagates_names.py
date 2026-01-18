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
def test_series_groupby_propagates_names():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    ddf = dd.from_pandas(df, 2)
    func = lambda df: df['y'].sum()
    with pytest.warns(UserWarning):
        result = ddf.groupby('x').apply(func, **INCLUDE_GROUPS)
    expected = df.groupby('x').apply(func, **INCLUDE_GROUPS)
    assert_eq(result, expected)