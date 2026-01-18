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
@pytest.mark.skipif(not PANDAS_GE_200, reason='dataframe.convert-string requires pandas>=2.0')
def test_read_parquet_convert_string_nullable_mapper(tmp_path, engine):
    """Make sure that when convert_string, dtype_backend and types_mapper are set,
    all three are used."""
    df = pd.DataFrame({'A': pd.Series(['def', 'abc', 'ghi'], dtype='string'), 'B': pd.Series([5, 2, 3], dtype='Int64'), 'C': pd.Series([1.1, 6.3, 8.4], dtype='Float32'), 'I': pd.Series(['x', 'y', 'z'], dtype='string')}).set_index('I')
    outfile = tmp_path / 'out.parquet'
    df.to_parquet(outfile, engine=engine)
    types_mapper = {pa.float32(): pd.Float64Dtype()}
    with dask.config.set({'dataframe.convert-string': True}):
        ddf = dd.read_parquet(tmp_path, engine='pyarrow', dtype_backend='numpy_nullable', arrow_to_pandas={'types_mapper': types_mapper.get})
    expected = df.astype({'A': 'string[pyarrow]', 'B': pd.Int64Dtype(), 'C': pd.Float64Dtype()})
    expected.index = expected.index.astype('string[pyarrow]')
    assert_eq(ddf, expected)