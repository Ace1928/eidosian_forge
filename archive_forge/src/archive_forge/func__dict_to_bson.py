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
def _dict_to_bson(doc: Any, check_keys: bool, opts: CodecOptions[Any], top_level: bool=True) -> bytes:
    """Encode a document to BSON."""
    if _raw_document_class(doc):
        return cast(bytes, doc.raw)
    try:
        elements = []
        if top_level and '_id' in doc:
            elements.append(_name_value_to_bson(b'_id\x00', doc['_id'], check_keys, opts))
        for key, value in doc.items():
            if not top_level or key != '_id':
                elements.append(_element_to_bson(key, value, check_keys, opts))
    except AttributeError:
        raise TypeError(f'encoder expected a mapping type but got: {doc!r}') from None
    encoded = b''.join(elements)
    return _PACK_INT(len(encoded) + 5) + encoded + b'\x00'