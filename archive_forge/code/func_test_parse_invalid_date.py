from __future__ import absolute_import
import copy
import datetime
import pickle
import hypothesis
import hypothesis.extra.pytz
import hypothesis.strategies
import pytest
from . import iso8601
@pytest.mark.parametrize('invalid_date, error_string', [('2013-10-', 'Unable to parse date string'), ('2013-', 'Unable to parse date string'), ('', 'Unable to parse date string'), ('wibble', 'Unable to parse date string'), ('23', 'Unable to parse date string'), ('131015T142533Z', 'Unable to parse date string'), ('131015', 'Unable to parse date string'), ('20141', 'Unable to parse date string'), ('201402', 'Unable to parse date string'), ('2007-06-23X06:40:34.00Z', 'Unable to parse date string'), ('2007-06-23 06:40:34.00Zrubbish', 'Unable to parse date string'), ('20114-01-03T01:45:49', 'Unable to parse date string')])
def test_parse_invalid_date(invalid_date: str, error_string: str) -> None:
    assert iso8601.is_iso8601(invalid_date) is False
    with pytest.raises(iso8601.ParseError) as exc:
        iso8601.parse_date(invalid_date)
    assert exc.errisinstance(iso8601.ParseError)
    assert str(exc.value).startswith(error_string)