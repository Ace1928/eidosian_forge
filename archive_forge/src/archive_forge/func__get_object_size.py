from __future__ import annotations
import datetime
import itertools
import os
import re
import struct
import sys
import uuid
from codecs import utf_8_decode as _utf_8_decode
from codecs import utf_8_encode as _utf_8_encode
from collections import abc as _abc
from typing import (
from bson.binary import (
from bson.code import Code
from bson.codec_options import (
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.errors import InvalidBSON, InvalidDocument, InvalidStringData
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
def _get_object_size(data: Any, position: int, obj_end: int) -> Tuple[int, int]:
    """Validate and return a BSON document's size."""
    try:
        obj_size = _UNPACK_INT_FROM(data, position)[0]
    except struct.error as exc:
        raise InvalidBSON(str(exc)) from None
    end = position + obj_size - 1
    if data[end] != 0:
        raise InvalidBSON('bad eoo')
    if end >= obj_end:
        raise InvalidBSON('invalid object length')
    if position == 0 and obj_size != obj_end:
        raise InvalidBSON('invalid object length')
    return (obj_size, end)