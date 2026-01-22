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
def _get_freq(ax: Axes, series: Series):
    freq = getattr(series.index, 'freq', None)
    if freq is None:
        freq = getattr(series.index, 'inferred_freq', None)
        freq = to_offset(freq, is_period=True)
    ax_freq = _get_ax_freq(ax)
    if freq is None:
        freq = ax_freq
    freq = _get_period_alias(freq)
    return (freq, ax_freq)