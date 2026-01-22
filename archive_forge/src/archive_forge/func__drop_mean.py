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
def _drop_mean(df, col=None):
    """TODO: In pandas 2.0, mean is implemented for datetimes, but Dask returns None."""
    if isinstance(df, pd.DataFrame):
        df.at['mean', col] = np.nan
        df.dropna(how='all', inplace=True)
    elif isinstance(df, pd.Series):
        df.drop(labels=['mean'], inplace=True, errors='ignore')
    else:
        raise NotImplementedError('Expected Series or DataFrame with mean')
    return df