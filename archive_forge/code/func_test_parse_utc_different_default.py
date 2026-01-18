from __future__ import absolute_import
import copy
import datetime
import pickle
import hypothesis
import hypothesis.extra.pytz
import hypothesis.strategies
import pytest
from . import iso8601
def test_parse_utc_different_default() -> None:
    """Z should mean 'UTC', not 'default'."""
    tz = iso8601.FixedOffset(2, 0, 'test offset')
    d = iso8601.parse_date('2007-01-01T08:00:00Z', default_timezone=tz)
    assert d == datetime.datetime(2007, 1, 1, 8, 0, 0, 0, iso8601.UTC)