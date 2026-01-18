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
def test_read_parquet_custom_columns(tmpdir, engine):
    tmp = str(tmpdir)
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32), 'f': np.arange(1000, dtype=np.float64)})
    df = dd.from_pandas(data, chunksize=50)
    df.to_parquet(tmp, engine=engine)
    df2 = dd.read_parquet(tmp, columns=['i32', 'f'], engine=engine, calculate_divisions=True)
    assert_eq(df[['i32', 'f']], df2, check_index=False)
    fns = glob.glob(os.path.join(tmp, '*.parquet'))
    df2 = dd.read_parquet(fns, columns=['i32'], engine=engine).compute()
    df2.sort_values('i32', inplace=True)
    assert_eq(df[['i32']], df2, check_index=False, check_divisions=False)
    df3 = dd.read_parquet(tmp, columns=['f', 'i32'], engine=engine, calculate_divisions=True)
    assert_eq(df[['f', 'i32']], df3, check_index=False)