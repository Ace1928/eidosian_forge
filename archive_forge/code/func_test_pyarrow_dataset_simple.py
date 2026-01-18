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
def test_pyarrow_dataset_simple(tmpdir, engine):
    fn = str(tmpdir)
    df = pd.DataFrame({'a': [4, 5, 6], 'b': ['a', 'b', 'b']})
    df = df.set_index('a', drop=True)
    ddf = dd.from_pandas(df, npartitions=2)
    ddf.to_parquet(fn, engine=engine)
    read_df = dd.read_parquet(fn, engine='pyarrow', calculate_divisions=True)
    read_df.compute()
    assert_eq(ddf, read_df)