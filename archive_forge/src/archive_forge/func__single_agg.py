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
def _single_agg(self, token, func, aggfunc=None, meta=None, split_every=None, split_out=1, shuffle_method=None, chunk_kwargs=None, aggregate_kwargs=None, columns=None):
    """
        Aggregation with a single function/aggfunc rather than a compound spec
        like in GroupBy.aggregate
        """
    shuffle_method = _determine_split_out_shuffle(shuffle_method, split_out)
    if aggfunc is None:
        aggfunc = func
    if chunk_kwargs is None:
        chunk_kwargs = {}
    if aggregate_kwargs is None:
        aggregate_kwargs = {}
    if meta is None:
        with check_numeric_only_deprecation():
            meta = func(self._meta_nonempty, **chunk_kwargs)
    if columns is None:
        columns = meta.name if is_series_like(meta) else meta.columns
    args = [self.obj] + (self.by if isinstance(self.by, list) else [self.by])
    token = self._token_prefix + token
    levels = _determine_levels(self.by)
    if shuffle_method:
        return _shuffle_aggregate(args, chunk=_apply_chunk, chunk_kwargs={'chunk': func, 'columns': columns, **self.observed, **self.dropna, **chunk_kwargs}, aggregate=_groupby_aggregate, aggregate_kwargs={'aggfunc': aggfunc, 'levels': levels, **self.observed, **self.dropna, **aggregate_kwargs}, token=token, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, sort=self.sort)
    return aca(args, chunk=_apply_chunk, chunk_kwargs=dict(chunk=func, columns=columns, **self.observed, **chunk_kwargs, **self.dropna), aggregate=_groupby_aggregate, meta=meta, token=token, split_every=split_every, aggregate_kwargs=dict(aggfunc=aggfunc, levels=levels, **self.observed, **aggregate_kwargs, **self.dropna), split_out=split_out, split_out_setup=split_out_on_index, sort=self.sort)