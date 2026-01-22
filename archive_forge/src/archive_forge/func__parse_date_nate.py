import re
from .w3dtf import _parse_date_w3dtf
def _parse_date_nate(dateString):
    """Parse a string according to the Nate 8-bit date format"""
    m = _korean_nate_date_re.match(dateString)
    if not m:
        return
    hour = int(m.group(5))
    ampm = m.group(4)
    if ampm == _korean_pm:
        hour += 12
    hour = str(hour)
    if len(hour) == 1:
        hour = '0' + hour
    w3dtfdate = '%(year)s-%(month)s-%(day)sT%(hour)s:%(minute)s:%(second)s%(zonediff)s' % {'year': m.group(1), 'month': m.group(2), 'day': m.group(3), 'hour': hour, 'minute': m.group(6), 'second': m.group(7), 'zonediff': '+09:00'}
    return _parse_date_w3dtf(w3dtfdate)