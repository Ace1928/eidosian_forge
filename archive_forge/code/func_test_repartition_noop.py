from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='testing this over in expr')
@pytest.mark.parametrize('type_ctor', [lambda o: o, tuple, list])
def test_repartition_noop(type_ctor):
    df = pd.DataFrame({'x': [1, 2, 4, 5], 'y': [6, 7, 8, 9]}, index=[-1, 0, 2, 7])
    ddf = dd.from_pandas(df, npartitions=2)
    ds = ddf.x
    ddf2 = ddf.repartition(divisions=type_ctor(ddf.divisions))
    assert ddf2 is ddf
    ddf3 = dd.repartition(ddf, divisions=type_ctor(ddf.divisions))
    assert ddf3 is ddf
    ds2 = ds.repartition(divisions=type_ctor(ds.divisions))
    assert ds2 is ds
    ds3 = dd.repartition(ds, divisions=type_ctor(ds.divisions))
    assert ds3 is ds