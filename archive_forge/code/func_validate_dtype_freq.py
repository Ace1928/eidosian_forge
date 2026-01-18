from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com
def validate_dtype_freq(dtype, freq: BaseOffsetT | BaseOffset | timedelta | str | None) -> BaseOffsetT:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
    freq : DateOffset or None

    Returns
    -------
    freq : DateOffset

    Raises
    ------
    ValueError : non-period dtype
    IncompatibleFrequency : mismatch between dtype and freq
    """
    if freq is not None:
        freq = to_offset(freq, is_period=True)
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if not isinstance(dtype, PeriodDtype):
            raise ValueError('dtype must be PeriodDtype')
        if freq is None:
            freq = dtype.freq
        elif freq != dtype.freq:
            raise IncompatibleFrequency('specified freq and dtype are different')
    return freq