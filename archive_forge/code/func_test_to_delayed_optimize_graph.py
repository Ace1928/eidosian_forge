from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
def test_to_delayed_optimize_graph():
    df = pd.DataFrame({'x': list(range(20))})
    ddf = dd.from_pandas(df, npartitions=20)
    ddf2 = (ddf + 1).loc[:2]
    d = ddf2.to_delayed()[0]
    assert len(d.dask) < 20
    d2 = ddf2.to_delayed(optimize_graph=False)[0]
    assert sorted(d2.dask) == sorted(ddf2.dask)
    assert_eq(ddf2.get_partition(0), d.compute())
    assert_eq(ddf2.get_partition(0), d2.compute())
    x = ddf2.x.sum()
    dx = x.to_delayed()
    dx2 = x.to_delayed(optimize_graph=False)
    assert len(dx.dask) < len(dx2.dask)
    assert_eq(dx.compute(), dx2.compute())