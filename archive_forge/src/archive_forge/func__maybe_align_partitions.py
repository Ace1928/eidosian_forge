from __future__ import annotations
import math
import pickle
import warnings
from functools import partial, wraps
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from tlz import merge_sorted, unique
from dask.base import is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
from dask.dataframe.dispatch import group_split_dispatch, hash_object_dispatch
from dask.dataframe.io import from_pandas
from dask.dataframe.shuffle import (
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.layers import BroadcastJoinLayer
from dask.utils import M, apply, get_default_shuffle_method
def _maybe_align_partitions(args):
    """Align DataFrame blocks if divisions are different.

    Note that if all divisions are unknown, but have equal npartitions, then
    they will be passed through unchanged. This is different than
    `align_partitions`, which will fail if divisions aren't all known"""
    _is_broadcastable = partial(is_broadcastable, args)
    dfs = [df for df in args if isinstance(df, _Frame) and (not _is_broadcastable(df))]
    if not dfs:
        return args
    divisions = dfs[0].divisions
    if not all((df.divisions == divisions for df in dfs)):
        dfs2 = iter(align_partitions(*dfs)[0])
        return [a if not isinstance(a, _Frame) else next(dfs2) for a in args]
    return args