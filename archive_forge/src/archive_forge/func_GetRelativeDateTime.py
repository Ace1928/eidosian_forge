from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
def GetRelativeDateTime(self, dt):
    """Returns a copy of the datetime object dt relative to the duration.

    Args:
      dt: The datetime object to add the duration to.

    Returns:
      The a copy of datetime object dt relative to the duration.
    """
    microsecond, second, minute, hour, day, month, year = (dt.microsecond, dt.second, dt.minute, dt.hour, dt.day, dt.month, dt.year)
    microsecond += self.microseconds
    if microsecond >= _MICROSECONDS_PER_SECOND:
        microsecond -= _MICROSECONDS_PER_SECOND
        second += 1
    elif microsecond < 0:
        microsecond += _MICROSECONDS_PER_SECOND
        second -= 1
    second += self.seconds
    if second >= _SECONDS_PER_MINUTE:
        second -= _SECONDS_PER_MINUTE
        minute += 1
    elif second < 0:
        second += _SECONDS_PER_MINUTE
        minute -= 1
    minute += self.minutes
    if minute >= _MINUTES_PER_HOUR:
        minute -= _MINUTES_PER_HOUR
        hour += 1
    elif minute < 0:
        minute += _MINUTES_PER_HOUR
        hour -= 1
    carry = int((hour + self.hours) / _HOURS_PER_DAY)
    hour += self.hours - carry * _HOURS_PER_DAY
    if hour < 0:
        hour += _HOURS_PER_DAY
        carry -= 1
    day += carry
    month += self.months
    if month > _MONTHS_PER_YEAR:
        month -= _MONTHS_PER_YEAR
        year += 1
    elif month < 1:
        month += _MONTHS_PER_YEAR
        year -= 1
    year += self.years
    day += self.days
    if day < 1:
        while day < 1:
            month -= 1
            if month < 1:
                month = _MONTHS_PER_YEAR
                year -= 1
            day += DaysInCalendarMonth(year, month)
    else:
        while True:
            days_in_month = DaysInCalendarMonth(year, month)
            if day <= days_in_month:
                break
            day -= days_in_month
            month += 1
            if month > _MONTHS_PER_YEAR:
                month = 1
                year += 1
    return datetime.datetime(year, month, day, hour, minute, second, microsecond, dt.tzinfo)