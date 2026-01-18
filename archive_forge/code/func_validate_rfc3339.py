import re
import calendar
import six
def validate_rfc3339(date_string):
    """
    Validates dates against RFC3339 datetime format
    Leap seconds are no supported.
    """
    m = RFC3339_REGEX.match(date_string)
    if m is None:
        return False
    year, month, day = map(int, m.groups())
    if not year:
        return False
    _, max_day = calendar.monthrange(year, month)
    if not 1 <= day <= max_day:
        return False
    return True