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
def _parse_canonical_maxkey(doc: Any) -> MaxKey:
    """Decode a JSON MaxKey to bson.max_key.MaxKey."""
    if type(doc['$maxKey']) is not int or doc['$maxKey'] != 1:
        raise TypeError('$maxKey value must be 1: %s', (doc,))
    if len(doc) != 1:
        raise TypeError(f'Bad $minKey, extra field(s): {doc}')
    return MaxKey()