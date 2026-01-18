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
def udgettext(self, domain: str, message: str) -> str:
    """Like ``ugettext()``, but look the message up in the specified
        domain.
        """
    return self._domains.get(domain, self).ugettext(message)