import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def unmarshall_time(tyme):
    """Unmarshall a datetime dict.

    .. versionchanged:: 1.5
       Drop leap second.

    .. versionchanged:: 1.6
       Added support for timezone information.
    """
    second = min(tyme['second'], _MAX_DATETIME_SEC)
    dt = datetime.datetime(day=tyme['day'], month=tyme['month'], year=tyme['year'], hour=tyme['hour'], minute=tyme['minute'], second=second, microsecond=tyme['microsecond'])
    tzname = tyme.get('tzname')
    if tzname:
        tzname = 'UTC' if tzname == 'UTC+00:00' else tzname
        if zoneinfo:
            tzinfo = zoneinfo.ZoneInfo(tzname)
            dt = dt.replace(tzinfo=tzinfo)
        else:
            tzinfo = pytz.timezone(tzname)
            dt = tzinfo.localize(dt)
    return dt