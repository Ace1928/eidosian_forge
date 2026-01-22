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
def _get_binary(data: Any, _view: Any, position: int, obj_end: int, opts: CodecOptions[Any], dummy1: Any) -> Tuple[Union[Binary, uuid.UUID], int]:
    """Decode a BSON binary to bson.binary.Binary or python UUID."""
    length, subtype = _UNPACK_LENGTH_SUBTYPE_FROM(data, position)
    position += 5
    if subtype == 2:
        length2 = _UNPACK_INT_FROM(data, position)[0]
        position += 4
        if length2 != length - 4:
            raise InvalidBSON("invalid binary (st 2) - lengths don't match!")
        length = length2
    end = position + length
    if length < 0 or end > obj_end:
        raise InvalidBSON('bad binary object length')
    if subtype in ALL_UUID_SUBTYPES:
        uuid_rep = opts.uuid_representation
        binary_value = Binary(data[position:end], subtype)
        if uuid_rep == UuidRepresentation.UNSPECIFIED or (subtype == UUID_SUBTYPE and uuid_rep != STANDARD) or (subtype == OLD_UUID_SUBTYPE and uuid_rep == STANDARD):
            return (binary_value, end)
        return (binary_value.as_uuid(uuid_rep), end)
    if subtype == 0:
        value = data[position:end]
    else:
        value = Binary(data[position:end], subtype)
    return (value, end)