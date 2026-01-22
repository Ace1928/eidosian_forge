from __future__ import annotations
from datetime import (
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.period import (
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import inherit_names
def _maybe_convert_timedelta(self, other) -> int | npt.NDArray[np.int64]:
    """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
    if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
        if isinstance(self.freq, Tick):
            delta = self._data._check_timedeltalike_freq_compat(other)
            return delta
    elif isinstance(other, BaseOffset):
        if other.base == self.freq.base:
            return other.n
        raise raise_on_incompatible(self, other)
    elif is_integer(other):
        assert isinstance(other, int)
        return other
    raise raise_on_incompatible(self, None)