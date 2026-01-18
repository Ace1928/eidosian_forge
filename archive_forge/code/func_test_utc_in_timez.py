import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
def test_utc_in_timez(monkeypatch):
    """Test that ``LazyRfc3339UtcTime`` is rendered as ``str`` using UTC timestamp."""
    utcoffset8_local_time_in_naive_utc = datetime.datetime(year=2020, month=1, day=1, hour=1, minute=23, second=45, tzinfo=datetime.timezone(datetime.timedelta(hours=8))).astimezone(datetime.timezone.utc).replace(tzinfo=None)

    class mock_datetime:

        @classmethod
        def utcnow(cls):
            return utcoffset8_local_time_in_naive_utc
    monkeypatch.setattr('datetime.datetime', mock_datetime)
    rfc3339_utc_time = str(cherrypy._cplogging.LazyRfc3339UtcTime())
    expected_time = '2019-12-31T17:23:45Z'
    assert rfc3339_utc_time == expected_time