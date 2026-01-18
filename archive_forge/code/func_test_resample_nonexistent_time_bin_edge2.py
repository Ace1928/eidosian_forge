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
def test_resample_nonexistent_time_bin_edge2(self):
    index = date_range(start='2017-10-10', end='2017-10-20', freq='1h')
    index = index.tz_localize('UTC').tz_convert('America/Sao_Paulo')
    df = DataFrame(data=list(range(len(index))), index=index)
    result = df.groupby(pd.Grouper(freq='1D')).count()
    expected = date_range(start='2017-10-09', end='2017-10-20', freq='D', tz='America/Sao_Paulo', nonexistent='shift_forward', inclusive='left')
    tm.assert_index_equal(result.index, expected)