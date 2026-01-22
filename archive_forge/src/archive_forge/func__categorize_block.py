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
def _categorize_block(df, categories, index):
    """Categorize a dataframe with given categories

    df: DataFrame
    categories: dict mapping column name to iterable of categories
    """
    df = df.copy()
    for col, vals in categories.items():
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.set_categories(vals)
        else:
            cat_dtype = categorical_dtype(meta=df[col], categories=vals, ordered=False)
            df[col] = df[col].astype(cat_dtype)
    if index is not None:
        if is_categorical_dtype(df.index):
            ind = df.index.set_categories(index)
        else:
            cat_dtype = categorical_dtype(meta=df.index, categories=index, ordered=False)
            ind = df.index.astype(dtype=cat_dtype)
        ind.name = df.index.name
        df.index = ind
    return df