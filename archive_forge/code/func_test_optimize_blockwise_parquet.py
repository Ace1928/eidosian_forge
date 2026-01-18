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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
def test_optimize_blockwise_parquet(tmpdir, engine):
    size = 40
    npartitions = 2
    tmp = str(tmpdir)
    df = pd.DataFrame({'a': np.arange(size, dtype=np.int32)})
    expect = dd.from_pandas(df, npartitions=npartitions)
    expect.to_parquet(tmp, engine=engine)
    ddf = dd.read_parquet(tmp, engine=engine, calculate_divisions=True)
    layers = ddf.__dask_graph__().layers
    assert len(layers) == 1
    assert isinstance(list(layers.values())[0], Blockwise)
    assert_eq(ddf, expect)
    ddf += 1
    expect += 1
    ddf += 10
    expect += 10
    layers = ddf.__dask_graph__().layers
    assert len(layers) == 3
    assert all((isinstance(layer, Blockwise) for layer in layers.values()))
    keys = [(ddf._name, i) for i in range(npartitions)]
    graph = optimize_blockwise(ddf.__dask_graph__(), keys)
    layers = graph.layers
    name = list(layers.keys())[0]
    assert len(layers) == 1
    assert isinstance(layers[name], Blockwise)
    assert_eq(ddf, expect)