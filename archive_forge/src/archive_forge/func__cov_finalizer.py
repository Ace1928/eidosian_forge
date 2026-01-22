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
def _cov_finalizer(df, cols, std=False):
    vals = []
    num_elements = len(list(it.product(cols, repeat=2)))
    num_cols = len(cols)
    vals = list(range(num_elements))
    col_idx_mapping = dict(zip(cols, range(num_cols)))
    for i, j in it.combinations_with_replacement(df[cols].columns, 2):
        x = col_idx_mapping[i]
        y = col_idx_mapping[j]
        idx = x + num_cols * y
        mul_col = f'{i}{j}'
        ni = df['%s-count' % i]
        nj = df['%s-count' % j]
        n = np.sqrt(ni * nj)
        div = n - 1
        div[div < 0] = 0
        val = (df[mul_col] - df[i] * df[j] / n).values[0] / div.values[0]
        if std:
            ii = f'{i}{i}'
            jj = f'{j}{j}'
            std_val_i = (df[ii] - df[i] ** 2 / ni).values[0] / div.values[0]
            std_val_j = (df[jj] - df[j] ** 2 / nj).values[0] / div.values[0]
            sqrt_val = np.sqrt(std_val_i * std_val_j)
            if sqrt_val == 0:
                val = np.nan
            else:
                val = val / sqrt_val
        vals[idx] = val
        if i != j:
            idx = num_cols * x + y
            vals[idx] = val
    level_1 = cols
    index = pd.MultiIndex.from_product([level_1, level_1])
    return pd.Series(vals, index=index)