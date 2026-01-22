from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
class _GroupByMixin(PandasObject, SelectionMixin):
    """
    Provide the groupby facilities.
    """
    _attributes: list[str]
    _selection: IndexLabel | None = None
    _groupby: GroupBy
    _timegrouper: TimeGrouper

    def __init__(self, *, parent: Resampler, groupby: GroupBy, key=None, selection: IndexLabel | None=None, include_groups: bool=False) -> None:
        assert isinstance(groupby, GroupBy), type(groupby)
        assert isinstance(parent, Resampler), type(parent)
        for attr in self._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection
        self.binner = parent.binner
        self.key = key
        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)
        self.ax = parent.ax
        self.obj = parent.obj
        self.include_groups = include_groups

    @no_type_check
    def _apply(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """

        def func(x):
            x = self._resampler_cls(x, timegrouper=self._timegrouper, gpr_index=self.ax)
            if isinstance(f, str):
                return getattr(x, f)(**kwargs)
            return x.apply(f, *args, **kwargs)
        result = _apply(self._groupby, func, include_groups=self.include_groups)
        return self._wrap_result(result)
    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

    @final
    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                assert subset.ndim == 1
        try:
            if isinstance(key, list) and self.key not in key and (self.key is not None):
                key.append(self.key)
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby
        selection = self._infer_selection(key, subset)
        new_rs = type(self)(groupby=groupby, parent=cast(Resampler, self), selection=selection)
        return new_rs