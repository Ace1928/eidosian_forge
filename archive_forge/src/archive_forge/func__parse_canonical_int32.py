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
def _parse_canonical_int32(doc: Any) -> int:
    """Decode a JSON int32 to python int."""
    i_str = doc['$numberInt']
    if len(doc) != 1:
        raise TypeError(f'Bad $numberInt, extra field(s): {doc}')
    if not isinstance(i_str, str):
        raise TypeError(f'$numberInt must be string: {doc}')
    return int(i_str)