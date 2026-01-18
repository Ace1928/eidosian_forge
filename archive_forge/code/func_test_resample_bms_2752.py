from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
def test_resample_bms_2752(self):
    timeseries = Series(index=pd.bdate_range('20000101', '20000201'), dtype=np.float64)
    res1 = timeseries.resample('BMS').mean()
    res2 = timeseries.resample('BMS').mean().resample('B').mean()
    assert res1.index[0] == Timestamp('20000103')
    assert res1.index[0] == res2.index[0]