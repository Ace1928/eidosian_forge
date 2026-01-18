import pprint
import pytest
import pytz  # noqa  # a test below uses pytz but only inside a `eval` call
from pandas import Timestamp
def test_repr_utcoffset(self):
    date_with_utc_offset = Timestamp('2014-03-13 00:00:00-0400', tz=None)
    assert '2014-03-13 00:00:00-0400' in repr(date_with_utc_offset)
    assert 'tzoffset' not in repr(date_with_utc_offset)
    assert 'UTC-04:00' in repr(date_with_utc_offset)
    expr = repr(date_with_utc_offset)
    assert date_with_utc_offset == eval(expr)