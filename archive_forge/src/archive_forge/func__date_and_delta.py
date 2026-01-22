from __future__ import annotations
import collections.abc
import datetime as dt
import math
import typing
from enum import Enum
from functools import total_ordering
from typing import Any
from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma
def _date_and_delta(value: typing.Any, *, now: dt.datetime | None=None) -> tuple[typing.Any, typing.Any]:
    """Turn a value into a date and a timedelta which represents how long ago it was.

    If that's not possible, return `(None, value)`.
    """
    if not now:
        now = _now()
    if isinstance(value, dt.datetime):
        date = value
        delta = now - value
    elif isinstance(value, dt.timedelta):
        date = now - value
        delta = value
    else:
        try:
            value = int(value)
            delta = dt.timedelta(seconds=value)
            date = now - delta
        except (ValueError, TypeError):
            return (None, value)
    return (date, _abs_timedelta(delta))