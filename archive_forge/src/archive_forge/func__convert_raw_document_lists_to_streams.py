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
def _convert_raw_document_lists_to_streams(document: Any) -> None:
    """Convert raw array of documents to a stream of BSON documents."""
    cursor = document.get('cursor')
    if not cursor:
        return
    for key in ('firstBatch', 'nextBatch'):
        batch = cursor.get(key)
        if not batch:
            continue
        data = _array_of_documents_to_buffer(batch)
        if data:
            cursor[key] = [data]
        else:
            cursor[key] = []