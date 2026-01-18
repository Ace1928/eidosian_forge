from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
def to_julian_date(self) -> npt.NDArray[np.float64]:
    """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """
    year = np.asarray(self.year)
    month = np.asarray(self.month)
    day = np.asarray(self.day)
    testarr = month < 3
    year[testarr] -= 1
    month[testarr] += 12
    return day + np.fix((153 * month - 457) / 5) + 365 * year + np.floor(year / 4) - np.floor(year / 100) + np.floor(year / 400) + 1721118.5 + (self.hour + self.minute / 60 + self.second / 3600 + self.microsecond / 3600 / 10 ** 6 + self.nanosecond / 3600 / 10 ** 9) / 24