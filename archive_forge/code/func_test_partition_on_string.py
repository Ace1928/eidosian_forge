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
@pytest.mark.parametrize('partition_on', ['aa', ['aa']])
def test_partition_on_string(tmpdir, partition_on):
    tmpdir = str(tmpdir)
    with dask.config.set(scheduler='single-threaded'):
        tmpdir = str(tmpdir)
        df = pd.DataFrame({'aa': np.random.choice(['A', 'B', 'C'], size=100), 'bb': np.random.random(size=100), 'cc': np.random.randint(1, 5, size=100)})
        d = dd.from_pandas(df, npartitions=2)
        d.to_parquet(tmpdir, partition_on=partition_on, write_index=False)
        out = dd.read_parquet(tmpdir, index=False, calculate_divisions=False)
    out = out.compute()
    for val in df.aa.unique():
        assert set(df.bb[df.aa == val]) == set(out.bb[out.aa == val])