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
def _json_convert(obj: Any, json_options: JSONOptions=DEFAULT_JSON_OPTIONS) -> Any:
    """Recursive helper method that converts BSON types so they can be
    converted into json.
    """
    if hasattr(obj, 'items'):
        return SON(((k, _json_convert(v, json_options)) for k, v in obj.items()))
    elif hasattr(obj, '__iter__') and (not isinstance(obj, (str, bytes))):
        return [_json_convert(v, json_options) for v in obj]
    try:
        return default(obj, json_options)
    except TypeError:
        return obj