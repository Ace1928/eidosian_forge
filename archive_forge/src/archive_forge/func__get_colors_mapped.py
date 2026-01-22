from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _get_colors_mapped(series, colors):
    unique = series.unique()
    mapped = dict(zip(unique, colors))
    return [mapped[v] for v in series.values]