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
def warn_dtype_mismatch(left, right, left_on, right_on):
    """Checks for merge column dtype mismatches and throws a warning (#4574)"""
    if not isinstance(left_on, list):
        left_on = [left_on]
    if not isinstance(right_on, list):
        right_on = [right_on]
    if all((col in left.columns for col in left_on)) and all((col in right.columns for col in right_on)):
        dtype_mism = [((lo, ro), left.dtypes[lo], right.dtypes[ro]) for lo, ro in zip(left_on, right_on) if not is_dtype_equal(left.dtypes[lo], right.dtypes[ro])]
        if dtype_mism:
            col_tb = asciitable(('Merge columns', 'left dtype', 'right dtype'), dtype_mism)
            warnings.warn('Merging dataframes with merge column data type mismatches: \n{}\nCast dtypes explicitly to avoid unexpected results.'.format(col_tb))