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
def test_partition_parallel_metadata(tmpdir, engine):
    tmp = str(tmpdir)
    d = pd.DataFrame({'a': np.random.rand(50), 'b': np.random.choice(['x', 'y', 'z'], size=50), 'c': np.random.choice(['x', 'y', 'z'], size=50)})
    d = dd.from_pandas(d, 2)
    d.to_parquet(tmp, partition_on=['b'], engine=engine, write_metadata_file=False)
    df = dd.read_parquet(tmp, engine=engine, calculate_divisions=True, metadata_task_size=1)
    assert set(df.b.cat.categories) == {'x', 'y', 'z'}