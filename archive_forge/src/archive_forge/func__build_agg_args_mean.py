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
def _build_agg_args_mean(result_column, func, input_column):
    int_sum = _make_agg_id('sum', input_column)
    int_count = _make_agg_id('count', input_column)
    return dict(chunk_funcs=[(int_sum, _apply_func_to_column, dict(column=input_column, func=M.sum)), (int_count, _apply_func_to_column, dict(column=input_column, func=M.count))], aggregate_funcs=[(col, _apply_func_to_column, dict(column=col, func=M.sum)) for col in (int_sum, int_count)], finalizer=(result_column, _finalize_mean, dict(sum_column=int_sum, count_column=int_count)))