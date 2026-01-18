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
@pytest.mark.parametrize('convert_string', [True, False])
@pytest.mark.skipif(not PANDAS_GE_200, reason='dataframe.convert-string requires pandas>=2.0')
def test_read_parquet_convert_string(tmp_path, convert_string, engine):
    df = pd.DataFrame({'A': ['def', 'abc', 'ghi'], 'B': [5, 2, 3], 'C': ['x', 'y', 'z']}).set_index('C')
    outfile = tmp_path / 'out.parquet'
    df.to_parquet(outfile, engine=engine)
    with dask.config.set({'dataframe.convert-string': convert_string}):
        ddf = dd.read_parquet(outfile, engine=engine)
    if convert_string and engine == 'pyarrow':
        expected = df.astype({'A': 'string[pyarrow]'})
        expected.index = expected.index.astype('string[pyarrow]')
    else:
        expected = df
    assert_eq(ddf, expected)
    if not DASK_EXPR_ENABLED:
        assert len(ddf.dask.layers) == 1
    with dask.config.set({'dataframe.convert-string': convert_string}):
        ddf1 = dd.read_parquet(outfile, engine='pyarrow')
    with dask.config.set({'dataframe.convert-string': not convert_string}):
        ddf2 = dd.read_parquet(outfile, engine='pyarrow')
    if not DASK_EXPR_ENABLED:
        assert ddf1._name != ddf2._name