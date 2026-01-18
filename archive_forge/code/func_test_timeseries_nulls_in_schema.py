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
@pytest.mark.parametrize('schema', ['infer', None])
def test_timeseries_nulls_in_schema(tmpdir, engine, schema):
    tmp_path = str(tmpdir.mkdir('files'))
    tmp_path = os.path.join(tmp_path, '../', 'files')
    ddf2 = dask.datasets.timeseries(start='2000-01-01', end='2000-01-03', freq='1h').reset_index().map_partitions(lambda x: x.loc[:5])
    ddf2 = ddf2.set_index('x').reset_index().persist()
    ddf2.name = ddf2.name.where(ddf2.timestamp == '2000-01-01', None)
    ddf2.to_parquet(tmp_path, engine=engine, write_metadata_file=False, schema=schema)
    ddf_read = dd.read_parquet(tmp_path, engine=engine)
    assert_eq(ddf_read, ddf2, check_divisions=False, check_index=False)