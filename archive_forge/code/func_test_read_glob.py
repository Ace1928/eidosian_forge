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
def test_read_glob(tmpdir, write_engine, read_engine):
    tmp_path = str(tmpdir)
    ddf.to_parquet(tmp_path, engine=write_engine)
    if os.path.exists(os.path.join(tmp_path, '_metadata')):
        os.unlink(os.path.join(tmp_path, '_metadata'))
    files = os.listdir(tmp_path)
    assert '_metadata' not in files
    ddf2 = dd.read_parquet(os.path.join(tmp_path, '*.parquet'), engine=read_engine, index='myindex', calculate_divisions=True)
    assert_eq(ddf, ddf2)