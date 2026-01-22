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
def _contains_index_name(df, columns_or_index):
    """
    Test whether ``columns_or_index`` contains a reference
    to the index of ``df

    This is the local (non-collection) version of
    ``dask.core.DataFrame._contains_index_name``.
    """

    def _is_index_level_reference(x, key):
        return x.index.name is not None and (np.isscalar(key) or isinstance(key, tuple)) and (key == x.index.name) and (key not in getattr(x, 'columns', ()))
    if isinstance(columns_or_index, list):
        return any((_is_index_level_reference(df, n) for n in columns_or_index))
    else:
        return _is_index_level_reference(df, columns_or_index)