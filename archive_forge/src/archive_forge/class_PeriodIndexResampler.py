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
class PeriodIndexResampler(DatetimeIndexResampler):
    ax: PeriodIndex

    @property
    def _resampler_for_grouping(self):
        warnings.warn('Resampling a groupby with a PeriodIndex is deprecated. Cast to DatetimeIndex before resampling instead.', FutureWarning, stacklevel=find_stack_level())
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self):
        if self.kind == 'timestamp':
            return super()._get_binner_for_time()
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        obj = super()._convert_obj(obj)
        if self._from_selection:
            msg = 'Resampling from level= or on= selection with a PeriodIndex is not currently supported, use .set_index(...) to explicitly set index'
            raise NotImplementedError(msg)
        if self.kind == 'timestamp':
            obj = obj.to_timestamp(how=self.convention)
        return obj

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        if self.kind == 'timestamp':
            return super()._downsample(how, **kwargs)
        orig_how = how
        how = com.get_cython_func(how) or how
        if orig_how != how:
            warn_alias_replacement(self, orig_how, how)
        ax = self.ax
        if is_subperiod(ax.freq, self.freq):
            return self._groupby_and_aggregate(how, **kwargs)
        elif is_superperiod(ax.freq, self.freq):
            if how == 'ohlc':
                return self._groupby_and_aggregate(how)
            return self.asfreq()
        elif ax.freq == self.freq:
            return self.asfreq()
        raise IncompatibleFrequency(f'Frequency {ax.freq} cannot be resampled to {self.freq}, as they are not sub or super periods')

    def _upsample(self, method, limit: int | None=None, fill_value=None):
        """
        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
        if self.kind == 'timestamp':
            return super()._upsample(method, limit=limit, fill_value=fill_value)
        ax = self.ax
        obj = self.obj
        new_index = self.binner
        memb = ax.asfreq(self.freq, how=self.convention)
        if method == 'asfreq':
            method = None
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        new_obj = _take_new_index(obj, indexer, new_index, axis=self.axis)
        return self._wrap_result(new_obj)