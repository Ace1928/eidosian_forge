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
@pytest.mark.parametrize('start, stop, right_boundary, left_boundary, drop', [(-1, None, False, False, [-1, -2]), (-1, None, False, True, [-2]), (None, 3, False, False, [3, 4]), (None, 3, True, False, [4]), (-0.5, None, False, False, [-1, -2]), (-0.5, None, False, True, [-1, -2]), (-1.5, None, False, True, [-2]), (None, 3.5, False, False, [4]), (None, 3.5, True, False, [4]), (None, 2.5, False, False, [3, 4])])
def test_with_boundary(start, stop, right_boundary, left_boundary, drop):
    x = np.array([-1, -2, 2, 4, 3])
    df = pd.DataFrame({'B': range(len(x))}, index=x)
    result = methods.boundary_slice(df, start, stop, right_boundary, left_boundary)
    expected = df.drop(drop)
    tm.assert_frame_equal(result, expected)