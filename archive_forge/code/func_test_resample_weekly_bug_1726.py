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
def test_resample_weekly_bug_1726(self):
    ind = date_range(start='8/6/2012', end='8/26/2012', freq='D')
    n = len(ind)
    data = [[x] * 5 for x in range(n)]
    df = DataFrame(data, columns=['open', 'high', 'low', 'close', 'vol'], index=ind)
    df.resample('W-MON', closed='left', label='left').first()