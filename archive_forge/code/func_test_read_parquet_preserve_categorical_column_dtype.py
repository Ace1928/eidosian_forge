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
@pytest.mark.skipif(not PANDAS_GE_200, reason='pd.Index does not support int32 before 2.0')
def test_read_parquet_preserve_categorical_column_dtype(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    outdir = tmp_path / 'out.parquet'
    df.to_parquet(outdir, partition_cols=['a'])
    ddf = dd.read_parquet(outdir)
    expected = pd.DataFrame({'b': ['x', 'y'], 'a': pd.Categorical(pd.Index([1, 2], dtype='int32'))}, index=[0, 0])
    assert_eq(ddf, expected)