from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat
def test_raises_on_non_datetimelike_index():
    xp = DataFrame()
    msg = "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'RangeIndex'"
    with pytest.raises(TypeError, match=msg):
        xp.resample('YE')