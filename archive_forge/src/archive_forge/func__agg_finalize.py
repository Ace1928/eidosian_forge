from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def _agg_finalize(df, aggregate_funcs, finalize_funcs, level, sort=False, arg=None, columns=None, is_series=False, **kwargs):
    df = _groupby_apply_funcs(df, funcs=aggregate_funcs, level=level, sort=sort, **kwargs)
    result = collections.OrderedDict()
    for result_column, func, finalize_kwargs in finalize_funcs:
        result[result_column] = func(df, **finalize_kwargs)
    result = df.__class__(result)
    if columns is not None:
        try:
            result = result[columns]
        except KeyError:
            pass
    if is_series and arg is not None and (not isinstance(arg, (list, dict))) and (result.ndim == 2):
        result = result[result.columns[0]]
    return result