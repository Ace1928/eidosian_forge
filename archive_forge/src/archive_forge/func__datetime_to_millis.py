from __future__ import annotations
import calendar
import datetime
import functools
from typing import Any, Union, cast
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions, DatetimeConversion
from bson.errors import InvalidBSON
from bson.tz_util import utc
def _datetime_to_millis(dtm: datetime.datetime) -> int:
    """Convert datetime to milliseconds since epoch UTC."""
    if dtm.utcoffset() is not None:
        dtm = dtm - dtm.utcoffset()
    return int(calendar.timegm(dtm.timetuple()) * 1000 + dtm.microsecond // 1000)