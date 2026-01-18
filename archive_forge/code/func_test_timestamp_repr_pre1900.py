import pprint
import pytest
import pytz  # noqa  # a test below uses pytz but only inside a `eval` call
from pandas import Timestamp
def test_timestamp_repr_pre1900(self):
    stamp = Timestamp('1850-01-01', tz='US/Eastern')
    repr(stamp)
    iso8601 = '1850-01-01 01:23:45.012345'
    stamp = Timestamp(iso8601, tz='US/Eastern')
    result = repr(stamp)
    assert iso8601 in result