from __future__ import annotations
import decimal
import gettext
import locale
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Iterable
from babel.core import Locale
from babel.dates import format_date, format_datetime, format_time, format_timedelta
from babel.numbers import (
def _locales_to_names(locales: Iterable[str | Locale] | str | Locale | None) -> list[str] | None:
    """Normalize a `locales` argument to a list of locale names.

    :param locales: the list of locales in order of preference (items in
                    this list can be either `Locale` objects or locale
                    strings)
    """
    if locales is None:
        return None
    if isinstance(locales, Locale):
        return [str(locales)]
    if isinstance(locales, str):
        return [locales]
    return [str(locale) for locale in locales]