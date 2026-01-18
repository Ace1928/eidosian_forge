from __future__ import annotations
import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.errors import PerformanceWarning
from tlz import partition
from dask.dataframe._compat import (
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
from dask.utils import _deprecated_kwarg
def value_counts_aggregate(x, total_length=None, sort=True, ascending=False, normalize=False, **groupby_kwargs):
    out = value_counts_combine(x, **groupby_kwargs)
    if normalize:
        out /= total_length if total_length is not None else out.sum()
    if sort:
        out = out.sort_values(ascending=ascending)
    if PANDAS_GE_200 and normalize:
        out.name = 'proportion'
    return out