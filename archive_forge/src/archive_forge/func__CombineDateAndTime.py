from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
@staticmethod
def _CombineDateAndTime(date, time, tzinfo):
    """Creates a datetime object from date and time objects.

    This is similar to the datetime.combine method, but its timezone
    calculations are designed to work with pytz.

    Arguments:
      date: a datetime.date object, in any timezone
      time: a datetime.time object, in any timezone
      tzinfo: a pytz timezone object, or None

    Returns:
      a datetime.datetime object, in the timezone 'tzinfo'
    """
    naive_result = datetime.datetime(date.year, date.month, date.day, time.hour, time.minute, time.second)
    if tzinfo is None:
        return naive_result
    try:
        return tzinfo.localize(naive_result, is_dst=None)
    except AmbiguousTimeError:
        return min(tzinfo.localize(naive_result, is_dst=True), tzinfo.localize(naive_result, is_dst=False))
    except NonExistentTimeError:
        while True:
            naive_result += datetime.timedelta(minutes=1)
            try:
                return tzinfo.localize(naive_result, is_dst=None)
            except NonExistentTimeError:
                pass