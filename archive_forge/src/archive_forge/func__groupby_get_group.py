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
def _groupby_get_group(df, by_key, get_key, columns):
    grouped = _groupby_raise_unaligned(df, by=by_key, convert_by_to_list=False)
    try:
        if is_dataframe_like(df):
            grouped = grouped[columns]
        return grouped.get_group(get_key)
    except KeyError:
        if is_dataframe_like(df):
            df = df[columns]
        return df.iloc[0:0]