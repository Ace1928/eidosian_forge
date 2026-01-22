import time as _time
import math as _math
import sys
from operator import index as _index
@staticmethod
def _name_from_offset(delta):
    if not delta:
        return 'UTC'
    if delta < timedelta(0):
        sign = '-'
        delta = -delta
    else:
        sign = '+'
    hours, rest = divmod(delta, timedelta(hours=1))
    minutes, rest = divmod(rest, timedelta(minutes=1))
    seconds = rest.seconds
    microseconds = rest.microseconds
    if microseconds:
        return f'UTC{sign}{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}'
    if seconds:
        return f'UTC{sign}{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'UTC{sign}{hours:02d}:{minutes:02d}'