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
@pytest.mark.parametrize('as_frame', [False, False])
def test_to_dask_array_raises(as_frame):
    s = pd.Series([1, 2, 3, 4, 5, 6], name='foo')
    a = dd.from_pandas(s, npartitions=2)
    if as_frame:
        a = a.to_frame()
    with pytest.raises(ValueError, match='4 != 2'):
        a.to_dask_array((1, 2, 3, 4))
    with pytest.raises(ValueError, match='Unexpected value'):
        a.to_dask_array(5)