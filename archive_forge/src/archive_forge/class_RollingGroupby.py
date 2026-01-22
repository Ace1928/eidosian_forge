from __future__ import annotations
import datetime
import warnings
from numbers import Integral
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.core.window import Rolling as pd_Rolling
from dask.array.core import normalize_arg
from dask.base import tokenize
from dask.blockwise import BlockwiseDepDict
from dask.dataframe import methods
from dask.dataframe._compat import check_axis_keyword_deprecation
from dask.dataframe.core import (
from dask.dataframe.io import from_pandas
from dask.dataframe.multi import _maybe_align_partitions
from dask.dataframe.utils import (
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import M, apply, derived_from, funcname, has_keyword
class RollingGroupby(Rolling):

    def __init__(self, groupby, window=None, min_periods=None, center=False, win_type=None, axis=0):
        self._groupby_kwargs = groupby._groupby_kwargs
        self._groupby_slice = groupby._slice
        obj = groupby.obj
        if self._groupby_slice is not None:
            if isinstance(self._groupby_slice, str):
                sliced_plus = [self._groupby_slice]
            else:
                sliced_plus = list(self._groupby_slice)
            if isinstance(groupby.by, str):
                sliced_plus.append(groupby.by)
            else:
                sliced_plus.extend(groupby.by)
            obj = obj[sliced_plus]
        super().__init__(obj, window=window, min_periods=min_periods, center=center, win_type=win_type, axis=axis)

    def _rolling_kwargs(self):
        kwargs = super()._rolling_kwargs()
        if kwargs.get('axis', None) in (0, 'index'):
            kwargs.pop('axis')
        return kwargs

    @staticmethod
    def pandas_rolling_method(df, rolling_kwargs, name, *args, groupby_kwargs=None, groupby_slice=None, **kwargs):
        groupby = df.groupby(**groupby_kwargs)
        if groupby_slice:
            groupby = groupby[groupby_slice]
        rolling = groupby.rolling(**rolling_kwargs)
        return getattr(rolling, name)(*args, **kwargs).sort_index(level=-1)

    def _call_method(self, method_name, *args, **kwargs):
        return super()._call_method(method_name, *args, groupby_kwargs=self._groupby_kwargs, groupby_slice=self._groupby_slice, **kwargs)