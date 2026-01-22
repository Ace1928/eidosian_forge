from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
class DatetimeConversion(int, enum.Enum):
    """Options for decoding BSON datetimes."""
    DATETIME = 1
    'Decode a BSON UTC datetime as a :class:`datetime.datetime`.\n\n    BSON UTC datetimes that cannot be represented as a\n    :class:`~datetime.datetime` will raise an :class:`OverflowError`\n    or a :class:`ValueError`.\n\n    .. versionadded 4.3\n    '
    DATETIME_CLAMP = 2
    'Decode a BSON UTC datetime as a :class:`datetime.datetime`, clamping\n    to :attr:`~datetime.datetime.min` and :attr:`~datetime.datetime.max`.\n\n    .. versionadded 4.3\n    '
    DATETIME_MS = 3
    'Decode a BSON UTC datetime as a :class:`~bson.datetime_ms.DatetimeMS`\n    object.\n\n    .. versionadded 4.3\n    '
    DATETIME_AUTO = 4
    'Decode a BSON UTC datetime as a :class:`datetime.datetime` if possible,\n    and a :class:`~bson.datetime_ms.DatetimeMS` if not.\n\n    .. versionadded 4.3\n    '