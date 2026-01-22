import re
import sys
from datetime import datetime, timedelta
from datetime import tzinfo as dt_tzinfo
from functools import lru_cache
from typing import (
from dateutil import tz
from arrow import locales
from arrow.constants import DEFAULT_LOCALE
from arrow.util import next_weekday, normalize_timestamp
def _parse_multiformat(self, string: str, formats: Iterable[str]) -> datetime:
    _datetime: Optional[datetime] = None
    for fmt in formats:
        try:
            _datetime = self.parse(string, fmt)
            break
        except ParserMatchError:
            pass
    if _datetime is None:
        supported_formats = ', '.join(formats)
        raise ParserError(f'Could not match input {string!r} to any of the following formats: {supported_formats}.')
    return _datetime