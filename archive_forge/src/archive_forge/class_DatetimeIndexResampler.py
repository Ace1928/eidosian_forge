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
class DatetimeIndexResampler(Resampler):
    ax: DatetimeIndex

    @property
    def _resampler_for_grouping(self):
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self):
        if self.kind == 'period':
            return self._timegrouper._get_time_period_bins(self.ax)
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        orig_how = how
        how = com.get_cython_func(how) or how
        if orig_how != how:
            warn_alias_replacement(self, orig_how, how)
        ax = self.ax
        obj = self._obj_with_exclusions
        if not len(ax):
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert obj.index.freq == self.freq, (obj.index.freq, self.freq)
            return obj
        if (ax.freq is not None or ax.inferred_freq is not None) and len(self._grouper.binlabels) > len(ax) and (how is None):
            return self.asfreq()
        if self.axis == 0:
            result = obj.groupby(self._grouper).aggregate(how, **kwargs)
        else:
            result = obj.T.groupby(self._grouper).aggregate(how, **kwargs).T
        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
        if self.closed == 'right':
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner

    def _upsample(self, method, limit: int | None=None, fill_value=None):
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
        if self.axis:
            raise AssertionError('axis must be 0')
        if self._from_selection:
            raise ValueError('Upsampling from level= or on= selection is not supported, use .set_index(...) to explicitly set index to datetime-like')
        ax = self.ax
        obj = self._selected_obj
        binner = self.binner
        res_index = self._adjust_binner_for_upsample(binner)
        if limit is None and to_offset(ax.inferred_freq) == self.freq and (len(obj) == len(res_index)):
            result = obj.copy()
            result.index = res_index
        else:
            if method == 'asfreq':
                method = None
            result = obj.reindex(res_index, method=method, limit=limit, fill_value=fill_value)
        return self._wrap_result(result)

    def _wrap_result(self, result):
        result = super()._wrap_result(result)
        if self.kind == 'period' and (not isinstance(result.index, PeriodIndex)):
            if isinstance(result.index, MultiIndex):
                if not isinstance(result.index.levels[-1], PeriodIndex):
                    new_level = result.index.levels[-1].to_period(self.freq)
                    result.index = result.index.set_levels(new_level, level=-1)
            else:
                result.index = result.index.to_period(self.freq)
        return result