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
def test_use_of_weakref_proxy():
    """Testing wrapping frames in proxy wrappers"""
    df = pd.DataFrame({'data': [1, 2, 3]})
    df_pxy = weakref.proxy(df)
    ser = pd.Series({'data': [1, 2, 3]})
    ser_pxy = weakref.proxy(ser)
    assert is_dataframe_like(df_pxy)
    assert is_series_like(ser_pxy)
    assert dask.dataframe.groupby._cov_chunk(df_pxy, 'data')
    assert isinstance(dask.dataframe.groupby._groupby_apply_funcs(df_pxy, 'data', funcs=[]), pd.DataFrame)
    l = []

    def f(x):
        l.append(x)
        return weakref.proxy(x)
    d = pd.DataFrame({'g': [0, 0, 1] * 3, 'b': [1, 2, 3] * 3})
    a = dd.from_pandas(d, npartitions=1)
    a = a.map_partitions(f, meta=a._meta)
    pxy = weakref.proxy(a)
    res = pxy['b'].groupby(pxy['g']).sum()
    isinstance(res.compute(), pd.Series)