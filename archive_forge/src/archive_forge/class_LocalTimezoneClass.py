import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
class LocalTimezoneClass(datetime.tzinfo):
    """This class defines local timezone."""
    ZERO = datetime.timedelta(0)
    HOUR = datetime.timedelta(hours=1)
    STDOFFSET = datetime.timedelta(seconds=-time.timezone)
    if time.daylight:
        DSTOFFSET = datetime.timedelta(seconds=-time.altzone)
    else:
        DSTOFFSET = STDOFFSET
    DSTDIFF = DSTOFFSET - STDOFFSET

    def utcoffset(self, dt):
        """datetime -> minutes east of UTC (negative for west of UTC)."""
        if self._isdst(dt):
            return self.DSTOFFSET
        else:
            return self.STDOFFSET

    def dst(self, dt):
        """datetime -> DST offset in minutes east of UTC."""
        if self._isdst(dt):
            return self.DSTDIFF
        else:
            return self.ZERO

    def tzname(self, dt):
        """datetime -> string name of time zone."""
        return time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        """Return true if given datetime is within local DST."""
        tt = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.weekday(), 0, -1)
        stamp = time.mktime(tt)
        tt = time.localtime(stamp)
        return tt.tm_isdst > 0

    def __repr__(self):
        """Return string '<Local>'."""
        return '<Local>'

    def localize(self, dt, unused_is_dst=False):
        """Convert naive time to local time."""
        if dt.tzinfo is not None:
            raise ValueError('Not naive datetime (tzinfo is already set)')
        return dt.replace(tzinfo=self)

    def normalize(self, dt, unused_is_dst=False):
        """Correct the timezone information on the given datetime."""
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')
        return dt.replace(tzinfo=self)