import re
from .rfc822 import _parse_date_rfc822
def _parse_date_greek(date_string):
    """Parse a string according to a Greek 8-bit date format."""
    m = _greek_date_format_re.match(date_string)
    if not m:
        return
    wday = _greek_wdays[m.group(1)]
    month = _greek_months[m.group(3)]
    rfc822date = '%(wday)s, %(day)s %(month)s %(year)s %(hour)s:%(minute)s:%(second)s %(zonediff)s' % {'wday': wday, 'day': m.group(2), 'month': month, 'year': m.group(4), 'hour': m.group(5), 'minute': m.group(6), 'second': m.group(7), 'zonediff': m.group(8)}
    return _parse_date_rfc822(rfc822date)