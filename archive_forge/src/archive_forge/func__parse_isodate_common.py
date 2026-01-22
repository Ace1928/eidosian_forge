from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
def _parse_isodate_common(self, dt_str):
    len_str = len(dt_str)
    components = [1, 1, 1]
    if len_str < 4:
        raise ValueError('ISO string too short')
    components[0] = int(dt_str[0:4])
    pos = 4
    if pos >= len_str:
        return (components, pos)
    has_sep = dt_str[pos:pos + 1] == self._DATE_SEP
    if has_sep:
        pos += 1
    if len_str - pos < 2:
        raise ValueError('Invalid common month')
    components[1] = int(dt_str[pos:pos + 2])
    pos += 2
    if pos >= len_str:
        if has_sep:
            return (components, pos)
        else:
            raise ValueError('Invalid ISO format')
    if has_sep:
        if dt_str[pos:pos + 1] != self._DATE_SEP:
            raise ValueError('Invalid separator in ISO string')
        pos += 1
    if len_str - pos < 2:
        raise ValueError('Invalid common day')
    components[2] = int(dt_str[pos:pos + 2])
    return (components, pos + 2)