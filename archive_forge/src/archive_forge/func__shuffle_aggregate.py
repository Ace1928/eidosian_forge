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
def _shuffle_aggregate(args, chunk=None, aggregate=None, token=None, chunk_kwargs=None, aggregate_kwargs=None, split_every=None, split_out=1, sort=True, ignore_index=False, shuffle_method='tasks'):
    """Shuffle-based groupby aggregation

    This algorithm may be more efficient than ACA for large ``split_out``
    values (required for high-cardinality groupby indices), but it also
    requires the output of ``chunk`` to be a proper DataFrame object.

    Parameters
    ----------
    args :
        Positional arguments for the `chunk` function. All `dask.dataframe`
        objects should be partitioned and indexed equivalently.
    chunk : function [block-per-arg] -> block
        Function to operate on each block of data
    aggregate : function concatenated-block -> block
        Function to operate on the concatenated result of chunk
    token : str, optional
        The name to use for the output keys.
    chunk_kwargs : dict, optional
        Keywords for the chunk function only.
    aggregate_kwargs : dict, optional
        Keywords for the aggregate function only.
    split_every : int, optional
        Number of partitions to aggregate into a shuffle partition.
        Defaults to eight, meaning that the initial partitions are repartitioned
        into groups of eight before the shuffle. Shuffling scales with the number
        of partitions, so it may be helpful to increase this number as a performance
        optimization, but only when the aggregated partition can comfortably
        fit in worker memory.
    split_out : int, optional
        Number of output partitions.
    ignore_index : bool, default False
        Whether the index can be ignored during the shuffle.
    sort : bool
        If allowed, sort the keys of the output aggregation.
    shuffle_method : str, default "tasks"
        Shuffle method to be used by ``DataFrame.shuffle``.
    """
    if chunk_kwargs is None:
        chunk_kwargs = dict()
    if aggregate_kwargs is None:
        aggregate_kwargs = dict()
    if not isinstance(args, (tuple, list)):
        args = [args]
    dfs = [arg for arg in args if isinstance(arg, _Frame)]
    npartitions = {arg.npartitions for arg in dfs}
    if len(npartitions) > 1:
        raise ValueError('All arguments must have same number of partitions')
    npartitions = npartitions.pop()
    if split_every is None:
        split_every = 8
    elif split_every is False:
        split_every = npartitions
    elif split_every < 1 or not isinstance(split_every, Integral):
        raise ValueError('split_every must be an integer >= 1')
    chunk_name = f'{token or funcname(chunk)}-chunk'
    chunked = map_partitions(chunk, *args, meta=chunk(*[arg._meta_nonempty if isinstance(arg, _Frame) else arg for arg in args], **chunk_kwargs), token=chunk_name, **chunk_kwargs)
    if is_series_like(chunked):
        series_name = chunked._meta.name
        chunked = chunked.to_frame('__series__')
        convert_back_to_series = True
    else:
        series_name = None
        convert_back_to_series = False
    shuffle_npartitions = max(chunked.npartitions // split_every, split_out)
    if sort is not None:
        aggregate_kwargs = aggregate_kwargs or {}
        aggregate_kwargs['sort'] = sort
    if sort is None and split_out > 1:
        idx = set(chunked._meta.columns) - set(chunked._meta.reset_index().columns)
        if len(idx) > 1:
            warnings.warn('In the future, `sort` for groupby operations will default to `True` to match the behavior of pandas. However, `sort=True` can have  significant performance implications when `split_out>1`. To avoid  global data shuffling, set `sort=False`.', FutureWarning)
    if sort and split_out > 1:
        cols = set(chunked.columns)
        chunked = chunked.reset_index()
        index_cols = sorted(set(chunked.columns) - cols)
        if len(index_cols) > 1:
            result = chunked.sort_values(index_cols, npartitions=shuffle_npartitions, shuffle_method=shuffle_method).map_partitions(M.set_index, index_cols, meta=chunked._meta.set_index(list(index_cols)), enforce_metadata=False)
        else:
            result = chunked.set_index(index_cols, npartitions=shuffle_npartitions, shuffle_method=shuffle_method)
    else:
        result = chunked.shuffle(chunked.index, ignore_index=ignore_index, npartitions=shuffle_npartitions, shuffle_method=shuffle_method)
    result = result.map_partitions(aggregate, **aggregate_kwargs)
    if convert_back_to_series:
        result = result['__series__'].rename(series_name)
    if split_out < shuffle_npartitions:
        return result.repartition(npartitions=split_out)
    return result