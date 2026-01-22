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
class DataFrameGroupBy(_GroupBy):
    _token_prefix = 'dataframe-groupby-'

    def __getitem__(self, key):
        with check_observed_deprecation():
            if isinstance(key, list):
                g = DataFrameGroupBy(self.obj, by=self.by, slice=key, sort=self.sort, **self.dropna, **self.observed)
            else:
                g = SeriesGroupBy(self.obj, by=self.by, slice=key, sort=self.sort, **self.dropna, **self.observed)
        if isinstance(key, tuple):
            key = list(key)
        g._meta = g._meta[key]
        return g

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) + list(filter(M.isidentifier, self.obj.columns))))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e) from e

    def _all_numeric(self):
        """Are all columns that we're not grouping on numeric?"""
        numerics = self.obj._meta._get_numeric_data()
        post_group_columns = self._meta.count().columns
        return len(set(post_group_columns) - set(numerics.columns)) == 0

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @_aggregate_docstring(based_on='pd.core.groupby.DataFrameGroupBy.aggregate')
    def aggregate(self, arg=None, split_every=None, split_out=1, shuffle_method=None, **kwargs):
        if arg == 'size':
            return self.size()
        return super().aggregate(arg=arg, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, **kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @_aggregate_docstring(based_on='pd.core.groupby.DataFrameGroupBy.agg')
    @numeric_only_not_implemented
    def agg(self, arg=None, split_every=None, split_out=1, shuffle_method=None, **kwargs):
        return self.aggregate(arg=arg, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, **kwargs)