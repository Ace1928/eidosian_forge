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
def test_optimize_getitem_and_nonblockwise(tmpdir, engine):
    path = os.path.join(tmpdir, 'path.parquet')
    df = pd.DataFrame({'a': [3, 4, 2], 'b': [1, 2, 4], 'c': [5, 4, 2], 'd': [1, 2, 3]}, index=['a', 'b', 'c'])
    df.to_parquet(path, engine=engine)
    df2 = dd.read_parquet(path, engine=engine)
    df2[['a', 'b']].rolling(3).max().compute()