from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_integer_dtype
from tlz import merge, merge_sorted, take
from dask.base import tokenize
from dask.dataframe.core import Series
from dask.dataframe.dispatch import tolist_dispatch
from dask.utils import is_cupy_type, random_state_data
def tree_width(N, to_binary=False):
    """Generate tree width suitable for ``merge_sorted`` given N inputs

    The larger N is, the more tasks are reduced in a single task.

    In theory, this is designed so all tasks are of comparable effort.
    """
    if N < 32:
        group_size = 2
    else:
        group_size = int(math.log(N))
    num_groups = N // group_size
    if to_binary or num_groups < 16:
        return 2 ** int(math.log(N / group_size, 2))
    else:
        return num_groups