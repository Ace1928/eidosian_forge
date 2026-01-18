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
def test_resample_anchored_intraday(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    df = DataFrame(rng.month, index=rng)
    result = df.resample('ME').mean()
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.resample('ME', kind='period').mean().to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index = expected.index.as_unit(unit)._with_freq('infer')
    assert expected.index.freq == 'ME'
    tm.assert_frame_equal(result, expected)
    result = df.resample('ME', closed='left').mean()
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        exp = df.shift(1, freq='D').resample('ME', kind='period').mean()
    exp = exp.to_timestamp(how='end')
    exp.index = exp.index + Timedelta(1, 'ns') - Timedelta(1, 'D')
    exp.index = exp.index.as_unit(unit)._with_freq('infer')
    assert exp.index.freq == 'ME'
    tm.assert_frame_equal(result, exp)