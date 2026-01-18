from __future__ import absolute_import
import copy
import datetime
import pickle
import hypothesis
import hypothesis.extra.pytz
import hypothesis.strategies
import pytest
from . import iso8601
@pytest.mark.parametrize('valid_date,expected_datetime,isoformat', [('2007-06-23 06:40:34.00Z', datetime.datetime(2007, 6, 23, 6, 40, 34, 0, iso8601.UTC), '2007-06-23T06:40:34+00:00'), ('1997-07-16T19:20+01:00', datetime.datetime(1997, 7, 16, 19, 20, 0, 0, iso8601.FixedOffset(1, 0, '+01:00')), '1997-07-16T19:20:00+01:00'), ('2007-01-01T08:00:00', datetime.datetime(2007, 1, 1, 8, 0, 0, 0, iso8601.UTC), '2007-01-01T08:00:00+00:00'), ('2006-10-20T15:34:56.123+02:30', datetime.datetime(2006, 10, 20, 15, 34, 56, 123000, iso8601.FixedOffset(2, 30, '+02:30')), None), ('2006-10-20T15:34:56Z', datetime.datetime(2006, 10, 20, 15, 34, 56, 0, iso8601.UTC), '2006-10-20T15:34:56+00:00'), ('2007-5-7T11:43:55.328Z', datetime.datetime(2007, 5, 7, 11, 43, 55, 328000, iso8601.UTC), '2007-05-07T11:43:55.328000+00:00'), ('2006-10-20T15:34:56.123Z', datetime.datetime(2006, 10, 20, 15, 34, 56, 123000, iso8601.UTC), '2006-10-20T15:34:56.123000+00:00'), ('2013-10-15T18:30Z', datetime.datetime(2013, 10, 15, 18, 30, 0, 0, iso8601.UTC), '2013-10-15T18:30:00+00:00'), ('2013-10-15T22:30+04', datetime.datetime(2013, 10, 15, 22, 30, 0, 0, iso8601.FixedOffset(4, 0, '+04:00')), '2013-10-15T22:30:00+04:00'), ('2013-10-15T1130-0700', datetime.datetime(2013, 10, 15, 11, 30, 0, 0, iso8601.FixedOffset(-7, 0, '-07:00')), '2013-10-15T11:30:00-07:00'), ('2013-10-15T1130+0700', datetime.datetime(2013, 10, 15, 11, 30, 0, 0, iso8601.FixedOffset(+7, 0, '+07:00')), '2013-10-15T11:30:00+07:00'), ('2013-10-15T1130+07', datetime.datetime(2013, 10, 15, 11, 30, 0, 0, iso8601.FixedOffset(+7, 0, '+07:00')), '2013-10-15T11:30:00+07:00'), ('2013-10-15T1130-07', datetime.datetime(2013, 10, 15, 11, 30, 0, 0, iso8601.FixedOffset(-7, 0, '-07:00')), '2013-10-15T11:30:00-07:00'), ('2013-10-15T15:00-03:30', datetime.datetime(2013, 10, 15, 15, 0, 0, 0, iso8601.FixedOffset(-3, -30, '-03:30')), '2013-10-15T15:00:00-03:30'), ('2013-10-15T183123Z', datetime.datetime(2013, 10, 15, 18, 31, 23, 0, iso8601.UTC), '2013-10-15T18:31:23+00:00'), ('2013-10-15T1831Z', datetime.datetime(2013, 10, 15, 18, 31, 0, 0, iso8601.UTC), '2013-10-15T18:31:00+00:00'), ('2013-10-15T18Z', datetime.datetime(2013, 10, 15, 18, 0, 0, 0, iso8601.UTC), '2013-10-15T18:00:00+00:00'), ('2013-10-15', datetime.datetime(2013, 10, 15, 0, 0, 0, 0, iso8601.UTC), '2013-10-15T00:00:00+00:00'), ('20131015T18:30Z', datetime.datetime(2013, 10, 15, 18, 30, 0, 0, iso8601.UTC), '2013-10-15T18:30:00+00:00'), ('2012-12-19T23:21:28.512400+00:00', datetime.datetime(2012, 12, 19, 23, 21, 28, 512400, iso8601.FixedOffset(0, 0, '+00:00')), '2012-12-19T23:21:28.512400+00:00'), ('2006-10-20T15:34:56.123+0230', datetime.datetime(2006, 10, 20, 15, 34, 56, 123000, iso8601.FixedOffset(2, 30, '+02:30')), '2006-10-20T15:34:56.123000+02:30'), ('19950204', datetime.datetime(1995, 2, 4, tzinfo=iso8601.UTC), '1995-02-04T00:00:00+00:00'), ('2010-07-20 15:25:52.520701+00:00', datetime.datetime(2010, 7, 20, 15, 25, 52, 520701, iso8601.FixedOffset(0, 0, '+00:00')), '2010-07-20T15:25:52.520701+00:00'), ('2010-06-12', datetime.datetime(2010, 6, 12, tzinfo=iso8601.UTC), '2010-06-12T00:00:00+00:00'), ('1985-04-12T23:20:50.52-05:30', datetime.datetime(1985, 4, 12, 23, 20, 50, 520000, iso8601.FixedOffset(-5, -30, '-05:30')), '1985-04-12T23:20:50.520000-05:30'), ('1997-08-29T06:14:00.000123Z', datetime.datetime(1997, 8, 29, 6, 14, 0, 123, iso8601.UTC), '1997-08-29T06:14:00.000123+00:00'), ('2014-02', datetime.datetime(2014, 2, 1, 0, 0, 0, 0, iso8601.UTC), '2014-02-01T00:00:00+00:00'), ('2014', datetime.datetime(2014, 1, 1, 0, 0, 0, 0, iso8601.UTC), '2014-01-01T00:00:00+00:00'), ('1997-08-29T06:14:00,000123Z', datetime.datetime(1997, 8, 29, 6, 14, 0, 123, iso8601.UTC), '1997-08-29T06:14:00.000123+00:00')])
def test_parse_valid_date(valid_date: str, expected_datetime: datetime.datetime, isoformat: str) -> None:
    assert iso8601.is_iso8601(valid_date) is True
    parsed = iso8601.parse_date(valid_date)
    assert parsed.year == expected_datetime.year
    assert parsed.month == expected_datetime.month
    assert parsed.day == expected_datetime.day
    assert parsed.hour == expected_datetime.hour
    assert parsed.minute == expected_datetime.minute
    assert parsed.second == expected_datetime.second
    assert parsed.microsecond == expected_datetime.microsecond
    assert parsed.tzinfo == expected_datetime.tzinfo
    assert parsed == expected_datetime
    assert parsed.isoformat() == expected_datetime.isoformat()
    copy.deepcopy(parsed)
    pickle.dumps(parsed)
    if isoformat:
        assert parsed.isoformat() == isoformat
    assert iso8601.parse_date(parsed.isoformat()) == parsed