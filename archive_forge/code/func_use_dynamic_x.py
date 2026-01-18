from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def use_dynamic_x(ax: Axes, data: DataFrame | Series) -> bool:
    freq = _get_index_freq(data.index)
    ax_freq = _get_ax_freq(ax)
    if freq is None:
        freq = ax_freq
    elif ax_freq is None and len(ax.get_lines()) > 0:
        return False
    if freq is None:
        return False
    freq_str = _get_period_alias(freq)
    if freq_str is None:
        return False
    if isinstance(data.index, ABCDatetimeIndex):
        freq_str = OFFSET_TO_PERIOD_FREQSTR.get(freq_str, freq_str)
        base = to_offset(freq_str, is_period=True)._period_dtype_code
        x = data.index
        if base <= FreqGroup.FR_DAY.value:
            return x[:1].is_normalized
        period = Period(x[0], freq_str)
        assert isinstance(period, Period)
        return period.to_timestamp().tz_localize(x.tz) == x[0]
    return True