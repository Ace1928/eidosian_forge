from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _get_dt_and_tzinfo(dt_or_tzinfo: _DtOrTzinfo) -> tuple[datetime.datetime | None, datetime.tzinfo]:
    """
    Parse a `dt_or_tzinfo` value into a datetime and a tzinfo.

    See the docs for this function's callers for semantics.

    :rtype: tuple[datetime, tzinfo]
    """
    if dt_or_tzinfo is None:
        dt = datetime.datetime.now()
        tzinfo = LOCALTZ
    elif isinstance(dt_or_tzinfo, str):
        dt = None
        tzinfo = get_timezone(dt_or_tzinfo)
    elif isinstance(dt_or_tzinfo, int):
        dt = None
        tzinfo = UTC
    elif isinstance(dt_or_tzinfo, (datetime.datetime, datetime.time)):
        dt = _get_datetime(dt_or_tzinfo)
        tzinfo = dt.tzinfo if dt.tzinfo is not None else UTC
    else:
        dt = None
        tzinfo = dt_or_tzinfo
    return (dt, tzinfo)