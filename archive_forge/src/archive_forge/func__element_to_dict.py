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
def _element_to_dict(data: Any, view: Any, position: int, obj_end: int, opts: CodecOptions[Any], raw_array: bool=False) -> Tuple[str, Any, int]:
    """Decode a single key, value pair."""
    element_type = data[position]
    position += 1
    element_name, position = _get_c_string(data, view, position, opts)
    if raw_array and element_type == ord(BSONARR):
        _, end = _get_object_size(data, position, len(data))
        return (element_name, view[position:end + 1], end + 1)
    try:
        value, position = _ELEMENT_GETTER[element_type](data, view, position, obj_end, opts, element_name)
    except KeyError:
        _raise_unknown_type(element_type, element_name)
    if opts.type_registry._decoder_map:
        custom_decoder = opts.type_registry._decoder_map.get(type(value))
        if custom_decoder is not None:
            value = custom_decoder(value)
    return (element_name, value, position)