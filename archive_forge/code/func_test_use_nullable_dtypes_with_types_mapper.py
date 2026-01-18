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
def test_use_nullable_dtypes_with_types_mapper(tmp_path, engine):
    df = pd.DataFrame({'a': pd.Series([1, 2, pd.NA, 3, 4], dtype='Int64'), 'b': pd.Series([True, pd.NA, False, True, False], dtype='boolean'), 'c': pd.Series([0.1, 0.2, 0.3, pd.NA, 0.4], dtype='Float64'), 'd': pd.Series(['a', 'b', 'c', 'd', pd.NA], dtype='string')})
    ddf = dd.from_pandas(df, npartitions=3)
    ddf.to_parquet(tmp_path, engine=engine)
    types_mapper = {pa.int64(): pd.Float32Dtype()}
    result = dd.read_parquet(tmp_path, engine='pyarrow', dtype_backend='numpy_nullable', arrow_to_pandas={'types_mapper': types_mapper.get})
    expected = df.astype({'a': pd.Float32Dtype()})
    if pyarrow_version.major >= 12:
        expected.index = expected.index.astype(pd.Float32Dtype())
    assert_eq(result, expected)