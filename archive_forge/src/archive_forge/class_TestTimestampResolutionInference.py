import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
class TestTimestampResolutionInference:

    def test_construct_from_time_unit(self):
        ts = Timestamp('01:01:01.111')
        assert ts.unit == 'ms'

    def test_constructor_str_infer_reso(self):
        ts = Timestamp('01/30/2023')
        assert ts.unit == 's'
        ts = Timestamp('2015Q1')
        assert ts.unit == 's'
        ts = Timestamp('2016-01-01 1:30:01 PM')
        assert ts.unit == 's'
        ts = Timestamp('2016 June 3 15:25:01.345')
        assert ts.unit == 'ms'
        ts = Timestamp('300-01-01')
        assert ts.unit == 's'
        ts = Timestamp('300 June 1:30:01.300')
        assert ts.unit == 'ms'
        ts = Timestamp('01-01-2013T00:00:00.000000000+0000')
        assert ts.unit == 'ns'
        ts = Timestamp('2016/01/02 03:04:05.001000 UTC')
        assert ts.unit == 'us'
        ts = Timestamp('01-01-2013T00:00:00.000000002100+0000')
        assert ts == Timestamp('01-01-2013T00:00:00.000000002+0000')
        assert ts.unit == 'ns'
        ts = Timestamp('2020-01-01 00:00+00:00')
        assert ts.unit == 's'
        ts = Timestamp('2020-01-01 00+00:00')
        assert ts.unit == 's'

    @pytest.mark.parametrize('method', ['now', 'today'])
    def test_now_today_unit(self, method):
        ts_from_method = getattr(Timestamp, method)()
        ts_from_string = Timestamp(method)
        assert ts_from_method.unit == ts_from_string.unit == 'us'