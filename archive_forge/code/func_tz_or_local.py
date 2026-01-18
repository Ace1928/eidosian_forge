from __future__ import annotations
import numbers
import os
import random
import sys
import time as _time
from calendar import monthrange
from datetime import date, datetime, timedelta
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from types import ModuleType
from typing import Any, Callable
from dateutil import tz as dateutil_tz
from dateutil.parser import isoparse
from kombu.utils.functional import reprcall
from kombu.utils.objects import cached_property
from .functional import dictfilter
from .text import pluralize
def tz_or_local(self, tzinfo: tzinfo | None=None) -> tzinfo:
    """Return either our local timezone or the provided timezone."""
    if tzinfo is None:
        return self.local
    return self.get_timezone(tzinfo)