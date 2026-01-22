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
def _cov_agg(_t, levels, ddof, std=False, sort=False):
    sums = []
    muls = []
    counts = []
    t = list(_t)
    cols = t[0][0].columns
    for x, mul, n, col_mapping in t:
        sums.append(x)
        muls.append(mul)
        counts.append(n)
        col_mapping = col_mapping
    total_sums = concat(sums).groupby(level=levels, sort=sort).sum()
    total_muls = concat(muls).groupby(level=levels, sort=sort).sum()
    total_counts = concat(counts).groupby(level=levels).sum()
    result = concat([total_sums, total_muls, total_counts], axis=1).groupby(level=levels).apply(_cov_finalizer, cols=cols, std=std)
    inv_col_mapping = {v: k for k, v in col_mapping.items()}
    idx_vals = result.index.names
    idx_mapping = list()
    if len(idx_vals) == 1 and all((n is None for n in idx_vals)):
        idx_vals = list(inv_col_mapping.keys() - set(total_sums.columns))
    for val in idx_vals:
        idx_name = inv_col_mapping.get(val, val)
        idx_mapping.append(idx_name)
        if len(result.columns.levels[0]) < len(col_mapping):
            try:
                col_mapping.pop(idx_name)
            except KeyError:
                pass
    keys = list(col_mapping.keys())
    for level in range(len(result.columns.levels)):
        result.columns = result.columns.set_levels(keys, level=level)
    result.index.set_names(idx_mapping, inplace=True)
    if PANDAS_GE_300:
        s_result = result.stack()
    else:
        s_result = result.stack(dropna=False)
    assert is_dataframe_like(s_result)
    return s_result