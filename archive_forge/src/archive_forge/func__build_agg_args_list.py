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
def _build_agg_args_list(result_column, func, input_column):
    intermediate = _make_agg_id('list', input_column)
    return dict(chunk_funcs=[(intermediate, _apply_func_to_column, dict(column=input_column, func=lambda s: s.apply(list)))], aggregate_funcs=[(intermediate, _apply_func_to_column, dict(column=intermediate, func=lambda s0: s0.apply(lambda chunks: list(it.chain.from_iterable(chunks)))))], finalizer=(result_column, itemgetter(intermediate), dict()))