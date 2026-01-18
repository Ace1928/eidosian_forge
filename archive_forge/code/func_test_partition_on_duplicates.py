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
def test_partition_on_duplicates(tmpdir, engine):
    tmpdir = str(tmpdir)
    df = pd.DataFrame({'a1': np.random.choice(['A', 'B', 'C'], size=100), 'a2': np.random.choice(['X', 'Y', 'Z'], size=100), 'data': np.random.random(size=100)})
    d = dd.from_pandas(df, npartitions=2)
    for _ in range(2):
        d.to_parquet(tmpdir, partition_on=['a1', 'a2'], engine=engine)
    out = dd.read_parquet(tmpdir, engine=engine).compute()
    assert len(df) == len(out)
    for _, _, files in os.walk(tmpdir):
        for file in files:
            assert file in ('part.0.parquet', 'part.1.parquet', '_common_metadata', '_metadata')