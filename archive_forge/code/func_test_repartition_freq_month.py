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
def test_repartition_freq_month():
    ts = pd.date_range('2015-01-01 00:00', '2015-05-01 23:50', freq='10min')
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(ts), 4)), columns=list('ABCD'), index=ts)
    ddf = dd.from_pandas(df, npartitions=1).repartition(freq='MS')
    assert_eq(df, ddf)
    assert ddf.divisions == (pd.Timestamp('2015-1-1 00:00:00'), pd.Timestamp('2015-2-1 00:00:00'), pd.Timestamp('2015-3-1 00:00:00'), pd.Timestamp('2015-4-1 00:00:00'), pd.Timestamp('2015-5-1 00:00:00'), pd.Timestamp('2015-5-1 23:50:00'))
    assert ddf.npartitions == 5