from datetime import datetime
import pprint
import dateutil.tz
import pytest
import pytz  # a test below uses pytz but only inside a `eval` call
from pandas import Timestamp
def test_repr_matches_pydatetime_tz_dateutil(self):
    utc = dateutil.tz.tzutc()
    dt_date = datetime(2013, 1, 2, tzinfo=utc)
    assert str(dt_date) == str(Timestamp(dt_date))
    dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=utc)
    assert str(dt_datetime) == str(Timestamp(dt_datetime))
    dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=utc)
    assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))