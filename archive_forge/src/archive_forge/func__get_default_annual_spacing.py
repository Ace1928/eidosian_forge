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
def _get_default_annual_spacing(nyears) -> tuple[int, int]:
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        min_spacing, maj_spacing = (1, 1)
    elif nyears < 20:
        min_spacing, maj_spacing = (1, 2)
    elif nyears < 50:
        min_spacing, maj_spacing = (1, 5)
    elif nyears < 100:
        min_spacing, maj_spacing = (5, 10)
    elif nyears < 200:
        min_spacing, maj_spacing = (5, 25)
    elif nyears < 600:
        min_spacing, maj_spacing = (10, 50)
    else:
        factor = nyears // 1000 + 1
        min_spacing, maj_spacing = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)