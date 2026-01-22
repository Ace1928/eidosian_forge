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
def _value_counts_aggregate(series_gb):
    data = {k: v.groupby(level=-1).sum() for k, v in series_gb}
    if not data:
        data = [pd.Series(index=series_gb.obj.index[:0], dtype='float64')]
    res = pd.concat(data, names=series_gb.obj.index.names)
    typed_levels = {i: res.index.levels[i].astype(series_gb.obj.index.levels[i].dtype) for i in range(len(res.index.levels))}
    res.index = res.index.set_levels(typed_levels.values(), level=typed_levels.keys(), verify_integrity=False)
    return res