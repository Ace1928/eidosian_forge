from __future__ import annotations
from collections import defaultdict
from numbers import Integral
import pandas as pd
from pandas.api.types import is_scalar
from tlz import partition_all
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.accessor import Accessor
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
def _get_categories(df, columns, index):
    res = {}
    for col in columns:
        x = df[col]
        if is_categorical_dtype(x):
            res[col] = x._constructor(x.cat.categories)
        else:
            res[col] = x.dropna().drop_duplicates()
    if index:
        if is_categorical_dtype(df.index):
            return (res, df.index.categories)
        return (res, df.index.dropna().drop_duplicates())
    return (res, None)