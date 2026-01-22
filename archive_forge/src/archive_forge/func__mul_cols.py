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
def _mul_cols(df, cols):
    """Internal function to be used with apply to multiply
    each column in a dataframe by every other column

    a b c -> a*a, a*b, b*b, b*c, c*c
    """
    _df = df.__class__()
    for i, j in it.combinations_with_replacement(cols, 2):
        col = f'{i}{j}'
        _df[col] = df[i] * df[j]
    _df.index = np.zeros(len(_df), dtype=int)
    return _df