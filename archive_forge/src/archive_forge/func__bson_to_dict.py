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
def _bson_to_dict(data: Any, opts: CodecOptions[_DocumentType]) -> _DocumentType:
    """Decode a BSON string to document_class."""
    data, view = get_data_and_view(data)
    try:
        if _raw_document_class(opts.document_class):
            return opts.document_class(data, opts)
        _, end = _get_object_size(data, 0, len(data))
        return cast('_DocumentType', _elements_to_dict(data, view, 4, end, opts))
    except InvalidBSON:
        raise
    except Exception:
        _, exc_value, exc_tb = sys.exc_info()
        raise InvalidBSON(str(exc_value)).with_traceback(exc_tb) from None