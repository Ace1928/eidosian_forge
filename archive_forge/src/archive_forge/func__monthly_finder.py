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
def _monthly_finder(vmin, vmax, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = _get_periods_per_ymd(freq)
    vmin_orig = vmin
    vmin, vmax = (int(vmin), int(vmax))
    span = vmax - vmin + 1
    info = np.zeros(span, dtype=[('val', int), ('maj', bool), ('min', bool), ('fmt', '|S8')])
    info['val'] = np.arange(vmin, vmax + 1)
    dates_ = info['val']
    info['fmt'] = ''
    year_start = (dates_ % 12 == 0).nonzero()[0]
    info_maj = info['maj']
    info_fmt = info['fmt']
    if span <= 1.15 * periodsperyear:
        info_maj[year_start] = True
        info['min'] = True
        info_fmt[:] = '%b'
        info_fmt[year_start] = '%b\n%Y'
        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = '%b\n%Y'
    elif span <= 2.5 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        info['fmt'][quarter_start] = True
        info['min'] = True
        info_fmt[quarter_start] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    elif span <= 4 * periodsperyear:
        info_maj[year_start] = True
        info['min'] = True
        jan_or_jul = (dates_ % 12 == 0) | (dates_ % 12 == 6)
        info_fmt[jan_or_jul] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    elif span <= 11 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        info['min'][quarter_start] = True
        info_fmt[year_start] = '%Y'
    else:
        nyears = span / periodsperyear
        min_anndef, maj_anndef = _get_default_annual_spacing(nyears)
        years = dates_[year_start] // 12 + 1
        major_idx = year_start[years % maj_anndef == 0]
        info_maj[major_idx] = True
        info['min'][year_start[years % min_anndef == 0]] = True
        info_fmt[major_idx] = '%Y'
    return info