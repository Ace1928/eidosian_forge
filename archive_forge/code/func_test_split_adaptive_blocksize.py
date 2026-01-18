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
@pytest.mark.parametrize('blocksize', [None, 1024, 4096, '1MiB'])
def test_split_adaptive_blocksize(tmpdir, blocksize, engine, metadata):
    nparts = 2
    df_size = 100
    row_group_size = 5
    df = pd.DataFrame({'a': np.random.choice(['apple', 'banana', 'carrot'], size=df_size), 'b': np.random.random(size=df_size), 'c': np.random.randint(1, 5, size=df_size), 'index': np.arange(0, df_size)}).set_index('index')
    ddf1 = dd.from_pandas(df, npartitions=nparts)
    ddf1.to_parquet(str(tmpdir), engine='pyarrow', row_group_size=row_group_size, write_metadata_file=metadata)
    if metadata:
        path = str(tmpdir)
    else:
        dirname = str(tmpdir)
        files = os.listdir(dirname)
        assert '_metadata' not in files
        path = os.path.join(dirname, '*.parquet')
    ddf2 = dd.read_parquet(path, engine=engine, blocksize=blocksize, split_row_groups='adaptive', calculate_divisions=True, index='index', aggregate_files=True)
    assert_eq(ddf1, ddf2, check_divisions=False)
    num_row_groups = df_size // row_group_size
    if not blocksize:
        assert ddf2.npartitions == ddf1.npartitions
    else:
        assert ddf2.npartitions < num_row_groups
        if blocksize == '1MiB':
            assert ddf2.npartitions == 1