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
def _groupby_apply_funcs(df, *by, **kwargs):
    """
    Group a dataframe and apply multiple aggregation functions.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to work on.
    by: list of groupers
        If given, they are added to the keyword arguments as the ``by``
        argument.
    funcs: list of result-colum, function, keywordargument triples
        The list of functions that are applied on the grouped data frame.
        Has to be passed as a keyword argument.
    kwargs:
        All keyword arguments, but ``funcs``, are passed verbatim to the groupby
        operation of the dataframe

    Returns
    -------
    aggregated:
        the aggregated dataframe.
    """
    if len(by):
        kwargs.update(by=list(by))
    funcs = kwargs.pop('funcs')
    grouped = _groupby_raise_unaligned(df, **kwargs)
    result = collections.OrderedDict()
    for result_column, func, func_kwargs in funcs:
        r = func(grouped, **func_kwargs)
        if isinstance(r, tuple):
            for idx, s in enumerate(r):
                result[f'{result_column}-{idx}'] = s
        else:
            result[result_column] = r
    if is_dataframe_like(df):
        return df.__class__(result)
    else:
        return df.head(0).to_frame().__class__(result)