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
def test_to_parquet_overwrite_adaptive_round_trip(tmpdir, engine):
    df = pd.DataFrame({'a': range(128)})
    ddf = dd.from_pandas(df, npartitions=8)
    path = os.path.join(str(tmpdir), 'path')
    ddf.to_parquet(path, engine=engine)
    ddf2 = dd.read_parquet(path, engine=engine, split_row_groups='adaptive').repartition(partition_size='1GB')
    path_new = os.path.join(str(tmpdir), 'path_new')
    ddf2.to_parquet(path_new, engine=engine, overwrite=True)
    ddf2.to_parquet(path_new, engine=engine, overwrite=True)
    assert_eq(ddf2, dd.read_parquet(path_new, engine=engine, split_row_groups=False))