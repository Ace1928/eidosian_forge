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
def test_quantile_for_possibly_unsorted_q():
    """check that quantile is giving correct answers even when quantile parameter, q, may be unsorted.

    See https://github.com/dask/dask/issues/4642.
    """
    A = da.arange(0, 101)
    ds = dd.from_dask_array(A)
    for q in [[0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 0.99], [0.75, 0.5, 0.25], [0.25, 0.99, 0.75, 0.5]]:
        r = ds.quantile(q).compute()
        assert_eq(r.loc[0.25], 25.0)
        assert_eq(r.loc[0.5], 50.0)
        assert_eq(r.loc[0.75], 75.0)
    r = ds.quantile([0.25]).compute()
    assert_eq(r.loc[0.25], 25.0)
    r = ds.quantile(0.25).compute()
    assert_eq(r, 25.0)