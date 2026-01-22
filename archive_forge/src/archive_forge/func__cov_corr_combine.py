from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def _cov_corr_combine(data_in, corr=False):
    data = {'sum': None, 'count': None, 'cov': None}
    if corr:
        data['m'] = None
    for k in data.keys():
        data[k] = [d[k] for d in data_in]
        data[k] = np.concatenate(data[k]).reshape((len(data[k]),) + data[k][0].shape)
    sums = np.nan_to_num(data['sum'])
    counts = data['count']
    cum_sums = np.cumsum(sums, 0)
    cum_counts = np.cumsum(counts, 0)
    s1 = cum_sums[:-1]
    s2 = sums[1:]
    n1 = cum_counts[:-1]
    n2 = counts[1:]
    with np.errstate(invalid='ignore'):
        d = s2 / n2 - s1 / n1
        C = np.nansum(n1 * n2 / (n1 + n2) * (d * d.transpose((0, 2, 1))), 0) + np.nansum(data['cov'], 0)
    out = {'sum': cum_sums[-1], 'count': cum_counts[-1], 'cov': C}
    if corr:
        nobs = np.where(cum_counts[-1], cum_counts[-1], np.nan)
        mu = cum_sums[-1] / nobs
        counts_na = np.where(counts, counts, np.nan)
        m = np.nansum(data['m'] + counts * (sums / counts_na - mu) ** 2, axis=0)
        out['m'] = m
    return out