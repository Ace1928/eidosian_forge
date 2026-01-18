from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def weighted_quantile(series, weights, q):
    series = series.sort_values()
    cumsum = weights.reindex(series.index).fillna(0).cumsum()
    cutoff = cumsum.iloc[-1] * q
    return series[cumsum >= cutoff].iloc[0]