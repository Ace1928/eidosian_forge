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
@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize('n1, freq1, n2, freq2', [(30, 's', 0.5, 'Min'), (60, 's', 1, 'Min'), (3600, 's', 1, 'h'), (60, 'Min', 1, 'h'), (21600, 's', 0.25, 'D'), (86400, 's', 1, 'D'), (43200, 's', 0.5, 'D'), (1440, 'Min', 1, 'D'), (12, 'h', 0.5, 'D'), (24, 'h', 1, 'D')])
def test_resample_equivalent_offsets(n1, freq1, n2, freq2, k, unit):
    n1_ = n1 * k
    n2_ = n2 * k
    dti = date_range('1991-09-05', '1991-09-12', freq=freq1).as_unit(unit)
    ser = Series(range(len(dti)), index=dti)
    result1 = ser.resample(str(n1_) + freq1).mean()
    result2 = ser.resample(str(n2_) + freq2).mean()
    tm.assert_series_equal(result1, result2)