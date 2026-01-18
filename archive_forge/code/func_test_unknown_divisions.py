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
def test_unknown_divisions():
    dsk = {('x', 0): pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), ('x', 1): pd.DataFrame({'a': [4, 5, 6], 'b': [3, 2, 1]}), ('x', 2): pd.DataFrame({'a': [7, 8, 9], 'b': [0, 0, 0]})}
    if DASK_EXPR_ENABLED:
        d = dd.repartition(pd.concat(dsk.values()), divisions=[0, 1, 2, 10]).clear_divisions()
    else:
        meta = make_meta({'a': 'i8', 'b': 'i8'}, parent_meta=pd.DataFrame())
        d = dd.DataFrame(dsk, 'x', meta, [None, None, None, None])
    full = d.compute(scheduler='sync')
    assert_eq(d.a.sum(), full.a.sum())
    assert_eq(d.a + d.b + 1, full.a + full.b + 1)