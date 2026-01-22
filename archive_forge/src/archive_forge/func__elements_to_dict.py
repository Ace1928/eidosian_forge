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
def _elements_to_dict(data: Any, view: Any, position: int, obj_end: int, opts: CodecOptions[Any], result: Any=None, raw_array: bool=False) -> Any:
    """Decode a BSON document into result."""
    if result is None:
        result = opts.document_class()
    end = obj_end - 1
    while position < end:
        key, value, position = _element_to_dict(data, view, position, obj_end, opts, raw_array=raw_array)
        result[key] = value
    if position != obj_end:
        raise InvalidBSON('bad object or element length')
    return result