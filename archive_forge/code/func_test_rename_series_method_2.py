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
def test_rename_series_method_2():
    s = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g'], name='x')
    ds = dd.from_pandas(s, 2)
    for is_sorted in [True, False]:
        res = ds.rename(lambda x: x ** 2, sorted_index=is_sorted)
        assert_eq(res, s.rename(lambda x: x ** 2))
        assert res.known_divisions == is_sorted
        res = ds.rename(s, sorted_index=is_sorted)
        assert_eq(res, s.rename(s))
        assert res.known_divisions == is_sorted
    with pytest.raises(ValueError):
        ds.rename(lambda x: -x, sorted_index=True).divisions
    assert_eq(ds.rename(lambda x: -x), s.rename(lambda x: -x))
    res = ds.rename(ds)
    assert_eq(res, s.rename(s))
    assert not res.known_divisions
    ds2 = ds.clear_divisions()
    res = ds2.rename(lambda x: x ** 2, sorted_index=True)
    assert_eq(res, s.rename(lambda x: x ** 2))
    assert not res.known_divisions
    if not DASK_EXPR_ENABLED:
        with pytest.warns(FutureWarning, match='inplace'):
            res = ds.rename(lambda x: x ** 2, inplace=True, sorted_index=True)
        assert res is ds
        s.rename(lambda x: x ** 2, inplace=True)
        assert_eq(ds, s)