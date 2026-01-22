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
def _get_time_period_bins(self, ax: DatetimeIndex):
    if not isinstance(ax, DatetimeIndex):
        raise TypeError(f'axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}')
    freq = self.freq
    if len(ax) == 0:
        binner = labels = PeriodIndex(data=[], freq=freq, name=ax.name, dtype=ax.dtype)
        return (binner, [], labels)
    labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)
    end_stamps = (labels + freq).asfreq(freq, 's').to_timestamp()
    if ax.tz:
        end_stamps = end_stamps.tz_localize(ax.tz)
    bins = ax.searchsorted(end_stamps, side='left')
    return (binner, bins, labels)