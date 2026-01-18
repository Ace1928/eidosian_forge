import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def utcnow_ts(microsecond=False):
    """Timestamp version of our utcnow function.

    See :py:class:`oslo_utils.fixture.TimeFixture`.

    .. versionchanged:: 1.3
       Added optional *microsecond* parameter.
    """
    if utcnow.override_time is None:
        timestamp = time.time()
        if not microsecond:
            timestamp = int(timestamp)
        return timestamp
    now = utcnow()
    timestamp = calendar.timegm(now.timetuple())
    if microsecond:
        timestamp += float(now.microsecond) / 1000000
    return timestamp