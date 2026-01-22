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
def _get_time(time: datetime.time | datetime.datetime | None, tzinfo: datetime.tzinfo | None=None) -> datetime.time:
    """
    Get a timezoned time from a given instant.

    .. warning:: The return values of this function may depend on the system clock.

    :param time: time, datetime or None
    :rtype: time
    """
    if time is None:
        time = datetime.datetime.now(UTC)
    elif isinstance(time, (int, float)):
        time = datetime.datetime.fromtimestamp(time, UTC)
    if time.tzinfo is None:
        time = time.replace(tzinfo=UTC)
    if isinstance(time, datetime.datetime):
        if tzinfo is not None:
            time = time.astimezone(tzinfo)
            if hasattr(tzinfo, 'normalize'):
                time = tzinfo.normalize(time)
        time = time.timetz()
    elif tzinfo is not None:
        time = time.replace(tzinfo=tzinfo)
    return time