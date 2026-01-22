from __future__ import annotations
import base64
import datetime
import json
import math
import re
import uuid
from typing import (
from bson.binary import ALL_UUID_SUBTYPES, UUID_SUBTYPE, Binary, UuidRepresentation
from bson.code import Code
from bson.codec_options import CodecOptions, DatetimeConversion
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
def _parse_canonical_datetime(doc: Any, json_options: JSONOptions) -> Union[datetime.datetime, DatetimeMS]:
    """Decode a JSON datetime to python datetime.datetime."""
    dtm = doc['$date']
    if len(doc) != 1:
        raise TypeError(f'Bad $date, extra field(s): {doc}')
    if isinstance(dtm, str):
        if dtm[-1] == 'Z':
            dt = dtm[:-1]
            offset = 'Z'
        elif dtm[-6] in ('+', '-') and dtm[-3] == ':':
            dt = dtm[:-6]
            offset = dtm[-6:]
        elif dtm[-5] in ('+', '-'):
            dt = dtm[:-5]
            offset = dtm[-5:]
        elif dtm[-3] in ('+', '-'):
            dt = dtm[:-3]
            offset = dtm[-3:]
        else:
            dt = dtm
            offset = ''
        dot_index = dt.rfind('.')
        microsecond = 0
        if dot_index != -1:
            microsecond = int(float(dt[dot_index:]) * 1000000)
            dt = dt[:dot_index]
        aware = datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S').replace(microsecond=microsecond, tzinfo=utc)
        if offset and offset != 'Z':
            if len(offset) == 6:
                hours, minutes = offset[1:].split(':')
                secs = int(hours) * 3600 + int(minutes) * 60
            elif len(offset) == 5:
                secs = int(offset[1:3]) * 3600 + int(offset[3:]) * 60
            elif len(offset) == 3:
                secs = int(offset[1:3]) * 3600
            if offset[0] == '-':
                secs *= -1
            aware = aware - datetime.timedelta(seconds=secs)
        if json_options.tz_aware:
            if json_options.tzinfo:
                aware = aware.astimezone(json_options.tzinfo)
            if json_options.datetime_conversion == DatetimeConversion.DATETIME_MS:
                return DatetimeMS(aware)
            return aware
        else:
            aware_tzinfo_none = aware.replace(tzinfo=None)
            if json_options.datetime_conversion == DatetimeConversion.DATETIME_MS:
                return DatetimeMS(aware_tzinfo_none)
            return aware_tzinfo_none
    return _millis_to_datetime(int(dtm), cast('CodecOptions[Any]', json_options))