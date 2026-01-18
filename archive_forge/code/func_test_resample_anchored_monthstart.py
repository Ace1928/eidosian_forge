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
@pytest.mark.parametrize('freq', ['MS', 'BMS', 'QS-MAR', 'YS-DEC', 'YS-JUN'])
def test_resample_anchored_monthstart(simple_date_range_series, freq, unit):
    ts = simple_date_range_series('1/1/2000', '12/31/2002')
    ts.index = ts.index.as_unit(unit)
    ts.resample(freq).mean()