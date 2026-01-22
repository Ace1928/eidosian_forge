import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def _parse_timedelta_part(match):
    unit = _resolve_unit(match.group('unit'))
    if not unit.endswith('s'):
        unit += 's'
    raw_value = match.group('value')
    if ':' in raw_value:
        return _parse_timedelta_composite(raw_value, unit)
    value = float(raw_value)
    if unit == 'months':
        unit = 'years'
        value = value / 12
    if unit == 'years':
        unit = 'days'
        value = value * days_per_year
    return _Saved_NS.derive(unit, value)