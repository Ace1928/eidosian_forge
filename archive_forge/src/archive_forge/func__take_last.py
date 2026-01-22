from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def _take_last(a, skipna=True):
    """
    take last row (Series) of DataFrame / last value of Series
    considering NaN.

    Parameters
    ----------
    a : pd.DataFrame or pd.Series
    skipna : bool, default True
        Whether to exclude NaN

    """

    def _last_valid(s):
        for i in range(1, min(10, len(s) + 1)):
            val = s.iloc[-i]
            if not pd.isnull(val):
                return val
        else:
            nonnull = s[s.notna()]
            if not nonnull.empty:
                return nonnull.iloc[-1]
        return None
    if skipna is False:
        return a.iloc[-1]
    elif is_dataframe_like(a):
        series_typ = type(a.iloc[0:1, 0])
        if a.empty:
            return series_typ([], dtype='float')
        return series_typ([_last_valid(a.iloc[:, i]) for i in range(len(a.columns))], index=a.columns)
    else:
        return _last_valid(a)