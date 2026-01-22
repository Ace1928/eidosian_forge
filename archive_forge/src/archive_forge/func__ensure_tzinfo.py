import math
import numbers
import re
import types
import warnings
from binascii import b2a_base64
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional, Union
from dateutil.parser import parse as _dateutil_parse
from dateutil.tz import tzlocal
def _ensure_tzinfo(dt: datetime) -> datetime:
    """Ensure a datetime object has tzinfo

    If no tzinfo is present, add tzlocal
    """
    if not dt.tzinfo:
        warnings.warn('Interpreting naive datetime as local %s. Please add timezone info to timestamps.' % dt, DeprecationWarning, stacklevel=4)
        dt = dt.replace(tzinfo=tzlocal())
    return dt