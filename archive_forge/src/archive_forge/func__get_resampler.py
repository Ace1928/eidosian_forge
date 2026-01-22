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
def _get_resampler(self, obj: NDFrame, kind=None) -> Resampler:
    """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : Series or DataFrame
        kind : string, optional
            'period','timestamp','timedelta' are valid

        Returns
        -------
        Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
    _, ax, _ = self._set_grouper(obj, gpr_index=None)
    if isinstance(ax, DatetimeIndex):
        return DatetimeIndexResampler(obj, timegrouper=self, kind=kind, axis=self.axis, group_keys=self.group_keys, gpr_index=ax)
    elif isinstance(ax, PeriodIndex) or kind == 'period':
        if isinstance(ax, PeriodIndex):
            warnings.warn('Resampling with a PeriodIndex is deprecated. Cast index to DatetimeIndex before resampling instead.', FutureWarning, stacklevel=find_stack_level())
        else:
            warnings.warn("Resampling with kind='period' is deprecated.  Use datetime paths instead.", FutureWarning, stacklevel=find_stack_level())
        return PeriodIndexResampler(obj, timegrouper=self, kind=kind, axis=self.axis, group_keys=self.group_keys, gpr_index=ax)
    elif isinstance(ax, TimedeltaIndex):
        return TimedeltaIndexResampler(obj, timegrouper=self, axis=self.axis, group_keys=self.group_keys, gpr_index=ax)
    raise TypeError(f"Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of '{type(ax).__name__}'")