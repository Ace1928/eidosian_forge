from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@PYARROW_MARK
@pytest.mark.parametrize('aggregate_files', ['a', 'b'])
def test_split_adaptive_aggregate_files(tmpdir, aggregate_files):
    blocksize = '1MiB'
    partition_on = ['a', 'b']
    df_size = 100
    df1 = pd.DataFrame({'a': np.random.choice(['apple', 'banana', 'carrot'], size=df_size), 'b': np.random.choice(['small', 'large'], size=df_size), 'c': np.random.random(size=df_size), 'd': np.random.randint(1, 100, size=df_size)})
    ddf1 = dd.from_pandas(df1, npartitions=9)
    ddf1.to_parquet(str(tmpdir), engine='pyarrow', partition_on=partition_on, write_index=False)
    if DASK_EXPR_ENABLED:
        ctx = contextlib.nullcontext()
    else:
        ctx = pytest.warns(FutureWarning, match='Behavior may change')
    with ctx:
        ddf2 = dd.read_parquet(str(tmpdir), engine='pyarrow', blocksize=blocksize, split_row_groups='adaptive', aggregate_files=aggregate_files)
    if aggregate_files == 'a':
        assert ddf2.npartitions == 3
    elif aggregate_files == 'b':
        assert ddf2.npartitions == 6
    df2 = ddf2.compute().sort_values(['c', 'd'])
    df1 = df1.sort_values(['c', 'd'])
    assert_eq(df1[['c', 'd']], df2[['c', 'd']], check_index=False)