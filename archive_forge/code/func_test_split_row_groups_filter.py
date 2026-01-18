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
def test_split_row_groups_filter(tmpdir, engine):
    tmp = str(tmpdir)
    df = pd.DataFrame({'i32': np.arange(800, dtype=np.int32), 'f': np.arange(800, dtype=np.float64)})
    df.index.name = 'index'
    search_val = 600
    filters = [('f', '==', search_val)]
    dd.from_pandas(df, npartitions=4).to_parquet(tmp, append=True, engine='pyarrow', row_group_size=50)
    ddf2 = dd.read_parquet(tmp, engine=engine)
    ddf3 = dd.read_parquet(tmp, engine=engine, calculate_divisions=True, split_row_groups=True, filters=filters)
    assert (ddf3['i32'] == search_val).any().compute()
    assert_eq(ddf2[ddf2['i32'] == search_val].compute(), ddf3[ddf3['i32'] == search_val].compute())