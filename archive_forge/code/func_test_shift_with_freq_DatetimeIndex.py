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
@pytest.mark.parametrize('data_freq,divs1', [('B', False), ('D', True), ('h', True)])
def test_shift_with_freq_DatetimeIndex(data_freq, divs1):
    df = _compat.makeTimeDataFrame()
    df = df.set_index(_compat.makeDateIndex(30, freq=data_freq))
    ddf = dd.from_pandas(df, npartitions=4)
    for freq, divs2 in [('s', True), ('W', False), (pd.Timedelta(10, unit='h'), True)]:
        for d, p in [(ddf, df), (ddf.A, df.A), (ddf.index, df.index)]:
            res = d.shift(2, freq=freq)
            assert_eq(res, p.shift(2, freq=freq))
            assert res.known_divisions == divs2
    res = ddf.index.shift(2)
    assert_eq(res, df.index.shift(2))
    assert res.known_divisions == divs1