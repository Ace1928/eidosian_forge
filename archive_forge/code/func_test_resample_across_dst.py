from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_across_dst():
    df1 = DataFrame([1477786980, 1477790580], columns=['ts'])
    dti1 = DatetimeIndex(pd.to_datetime(df1.ts, unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid'))
    df2 = DataFrame([1477785600, 1477789200], columns=['ts'])
    dti2 = DatetimeIndex(pd.to_datetime(df2.ts, unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid'), freq='h')
    df = DataFrame([5, 5], index=dti1)
    result = df.resample(rule='h').sum()
    expected = DataFrame([5, 5], index=dti2)
    tm.assert_frame_equal(result, expected)