from __future__ import annotations
import contextlib
import datetime as pydt
from datetime import (
import functools
from typing import (
import warnings
import matplotlib.dates as mdates
from matplotlib.ticker import (
from matplotlib.transforms import nonsingular
import matplotlib.units as munits
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._typing import (
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
import pandas.core.tools.datetimes as tools
class TimeSeries_DateFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    """
    axis: Axis

    def __init__(self, freq: BaseOffset, minor_locator: bool=False, dynamic_mode: bool=True, plot_obj=None) -> None:
        freq = to_offset(freq, is_period=True)
        self.format = None
        self.freq = freq
        self.locs: list[Any] = []
        self.formatdict: dict[Any, Any] | None = None
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _set_default_format(self, vmin, vmax):
        """Returns the default ticks spacing."""
        info = self.finder(vmin, vmax, self.freq)
        if self.isminor:
            format = np.compress(info['min'] & np.logical_not(info['maj']), info)
        else:
            format = np.compress(info['maj'], info)
        self.formatdict = {x: f for x, _, _, f in format}
        return self.formatdict

    def set_locs(self, locs) -> None:
        """Sets the locations of the ticks"""
        self.locs = locs
        vmin, vmax = tuple(self.axis.get_view_interval())
        if vmax < vmin:
            vmin, vmax = (vmax, vmin)
        self._set_default_format(vmin, vmax)

    def __call__(self, x, pos: int | None=0) -> str:
        if self.formatdict is None:
            return ''
        else:
            fmt = self.formatdict.pop(x, '')
            if isinstance(fmt, np.bytes_):
                fmt = fmt.decode('utf-8')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Period with BDay freq is deprecated', category=FutureWarning)
                period = Period(ordinal=int(x), freq=self.freq)
            assert isinstance(period, Period)
            return period.strftime(fmt)