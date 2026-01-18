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
@pytest.mark.parametrize('scheduler', [None, 'processes'])
def test_null_partition_pyarrow(tmpdir, scheduler):
    df = pd.DataFrame({'id': pd.Series([0, 1, None], dtype='Int64'), 'x': pd.Series([1, 2, 3], dtype='Int64')})
    ddf = dd.from_pandas(df, npartitions=1)
    ddf.to_parquet(str(tmpdir), partition_on='id')
    fns = glob.glob(os.path.join(tmpdir, 'id=*/*.parquet'))
    assert len(fns) == 3
    ddf_read = dd.read_parquet(str(tmpdir), dtype_backend='numpy_nullable', dataset={'partitioning': {'flavor': 'hive', 'schema': pa.schema([('id', pa.int64())])}})
    if pyarrow_version.major >= 12:
        ddf.index = ddf.index.astype('Int64')
    assert_eq(ddf[['x', 'id']], ddf_read[['x', 'id']], check_divisions=False, scheduler=scheduler)