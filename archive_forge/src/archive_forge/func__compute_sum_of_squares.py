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
def _compute_sum_of_squares(grouped, column):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'DataFrameGroupBy.grouper is deprecated and will be removed in a future version of pandas.', FutureWarning)
        if hasattr(grouped, 'grouper'):
            keys = grouped.grouper
        elif hasattr(grouped, '_grouper'):
            keys = grouped._grouper
        else:
            keys = grouped.grouping.keys
    df = grouped.obj[column].pow(2) if column else grouped.obj.pow(2)
    return df.groupby(keys).sum()