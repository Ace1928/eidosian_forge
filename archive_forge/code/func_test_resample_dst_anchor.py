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
def test_resample_dst_anchor(unit):
    dti = DatetimeIndex([datetime(2012, 11, 4, 23)], tz='US/Eastern').as_unit(unit)
    df = DataFrame([5], index=dti)
    dti = DatetimeIndex(df.index.normalize(), freq='D').as_unit(unit)
    expected = DataFrame([5], index=dti)
    tm.assert_frame_equal(df.resample(rule='D').sum(), expected)
    df.resample(rule='MS').sum()
    tm.assert_frame_equal(df.resample(rule='MS').sum(), DataFrame([5], index=DatetimeIndex([datetime(2012, 11, 1)], tz='US/Eastern', freq='MS').as_unit(unit)))