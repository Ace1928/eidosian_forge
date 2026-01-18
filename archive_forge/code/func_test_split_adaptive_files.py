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
@pytest.mark.parametrize('metadata', [True, False])
@pytest.mark.parametrize('partition_on', [None, 'a'])
@pytest.mark.parametrize('blocksize', [4096, '1MiB'])
def test_split_adaptive_files(tmpdir, blocksize, partition_on, metadata):
    df_size = 100
    df1 = pd.DataFrame({'a': np.random.choice(['apple', 'banana', 'carrot'], size=df_size), 'b': np.random.random(size=df_size), 'c': np.random.randint(1, 5, size=df_size)})
    ddf1 = dd.from_pandas(df1, npartitions=9)
    ddf1.to_parquet(str(tmpdir), engine='pyarrow', partition_on=partition_on, write_metadata_file=metadata, write_index=False)
    aggregate_files = partition_on if partition_on else True
    if isinstance(aggregate_files, str):
        if DASK_EXPR_ENABLED:
            ctx = contextlib.nullcontext()
        else:
            ctx = pytest.warns(FutureWarning, match='Behavior may change')
        with ctx:
            ddf2 = dd.read_parquet(str(tmpdir), engine='pyarrow', blocksize=blocksize, split_row_groups='adaptive', aggregate_files=aggregate_files)
    else:
        ddf2 = dd.read_parquet(str(tmpdir), engine='pyarrow', blocksize=blocksize, split_row_groups='adaptive', aggregate_files=aggregate_files)
    if blocksize == 4096:
        assert ddf2.npartitions < ddf1.npartitions
    elif blocksize == '1MiB':
        if partition_on:
            assert ddf2.npartitions == 3
        else:
            assert ddf2.npartitions == 1
    if partition_on:
        df2 = ddf2.compute().sort_values(['b', 'c'])
        df1 = df1.sort_values(['b', 'c'])
        assert_eq(df1[['b', 'c']], df2[['b', 'c']], check_index=False)
    else:
        assert_eq(ddf1, ddf2, check_divisions=False, check_index=False)