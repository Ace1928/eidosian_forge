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
@pytest.mark.parametrize('drop', [0, 9])
def test_slice_on_filtered_boundary(drop):
    x = np.arange(10)
    x[[5, 6]] -= 2
    df = pd.DataFrame({'A': x, 'B': np.arange(len(x))})
    pdf = df.set_index('A').query(f'B != {drop}')
    ddf = dd.from_pandas(df, 1).set_index('A').query(f'B != {drop}')
    result = dd.concat([ddf, ddf.rename(columns={'B': 'C'})], axis=1)
    expected = pd.concat([pdf, pdf.rename(columns={'B': 'C'})], axis=1)
    assert_eq(result, expected)