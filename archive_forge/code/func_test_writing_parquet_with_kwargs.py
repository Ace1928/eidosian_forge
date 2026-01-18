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
def test_writing_parquet_with_kwargs(tmpdir, engine):
    fn = str(tmpdir)
    path1 = os.path.join(fn, 'normal')
    path2 = os.path.join(fn, 'partitioned')
    df = pd.DataFrame({'a': np.random.choice(['A', 'B', 'C'], size=100), 'b': np.random.random(size=100), 'c': np.random.randint(1, 5, size=100)})
    df.index.name = 'index'
    ddf = dd.from_pandas(df, npartitions=3)
    write_kwargs = {'pyarrow': {'compression': 'snappy', 'coerce_timestamps': None, 'use_dictionary': True}, 'fastparquet': {'compression': 'snappy', 'times': 'int64', 'fixed_text': None}}
    ddf.to_parquet(path1, engine=engine, **write_kwargs[engine])
    out = dd.read_parquet(path1, engine=engine, calculate_divisions=True)
    assert_eq(out, ddf, check_index=engine != 'fastparquet')
    with dask.config.set(scheduler='sync'):
        ddf.to_parquet(path2, engine=engine, partition_on=['a'], **write_kwargs[engine])
    out = dd.read_parquet(path2, engine=engine).compute()
    for val in df.a.unique():
        assert set(df.b[df.a == val]) == set(out.b[out.a == val])