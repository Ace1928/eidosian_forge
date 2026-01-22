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
def _adjust_bin_edges(self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]:
    if self.freq.name in ('BME', 'ME', 'W') or self.freq.name.split('-')[0] in ('BQE', 'BYE', 'QE', 'YE', 'W'):
        if self.closed == 'right':
            edges_dti = binner.tz_localize(None)
            edges_dti = edges_dti + Timedelta(days=1, unit=edges_dti.unit).as_unit(edges_dti.unit) - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
            bin_edges = edges_dti.tz_localize(binner.tz).asi8
        else:
            bin_edges = binner.asi8
        if bin_edges[-2] > ax_values.max():
            bin_edges = bin_edges[:-1]
            binner = binner[:-1]
    else:
        bin_edges = binner.asi8
    return (binner, bin_edges)