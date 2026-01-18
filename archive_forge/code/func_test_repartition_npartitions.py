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
@pytest.mark.parametrize('use_index', [True, False])
@pytest.mark.parametrize('n', [1, 2, 4, 5])
@pytest.mark.parametrize('k', [1, 2, 4, 5])
@pytest.mark.parametrize('dtype', [float, 'M8[ns]'])
@pytest.mark.parametrize('transform', [lambda df: df, lambda df: df.x])
def test_repartition_npartitions(use_index, n, k, dtype, transform):
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6] * 10, 'y': list('abdabd') * 10}, index=pd.Series(list(range(0, 30)) * 2, dtype=dtype))
    df = transform(df)
    a = dd.from_pandas(df, npartitions=n, sort=use_index)
    b = a.repartition(npartitions=k)
    assert_eq(a, b)
    assert b.npartitions == k
    assert all(map(len, b.partitions))