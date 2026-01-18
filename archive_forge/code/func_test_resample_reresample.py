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
def test_resample_reresample(unit):
    dti = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='D').as_unit(unit)
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    bs = s.resample('B', closed='right', label='right').mean()
    result = bs.resample('8h').mean()
    assert len(result) == 25
    assert isinstance(result.index.freq, offsets.DateOffset)
    assert result.index.freq == offsets.Hour(8)