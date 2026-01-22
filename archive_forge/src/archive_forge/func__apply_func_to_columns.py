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
def _apply_func_to_columns(df_like, prefix, func):
    if is_dataframe_like(df_like):
        columns = df_like.columns
    else:
        columns = df_like.obj.columns
    columns = sorted((col for col in columns if col.startswith(prefix)))
    columns = [df_like[col] for col in columns]
    return func(*columns)