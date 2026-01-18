import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_strftime_dt64_days(self):
    ser = Series(date_range('20130101', periods=5))
    ser.iloc[0] = pd.NaT
    result = ser.dt.strftime('%Y/%m/%d')
    expected = Series([np.nan, '2013/01/02', '2013/01/03', '2013/01/04', '2013/01/05'])
    tm.assert_series_equal(result, expected)
    datetime_index = date_range('20150301', periods=5)
    result = datetime_index.strftime('%Y/%m/%d')
    expected = Index(['2015/03/01', '2015/03/02', '2015/03/03', '2015/03/04', '2015/03/05'], dtype=np.object_)
    tm.assert_index_equal(result, expected)