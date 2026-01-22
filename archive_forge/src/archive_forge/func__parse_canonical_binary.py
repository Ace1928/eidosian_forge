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
def _parse_canonical_binary(doc: Any, json_options: JSONOptions) -> Union[Binary, uuid.UUID]:
    binary = doc['$binary']
    b64 = binary['base64']
    subtype = binary['subType']
    if not isinstance(b64, str):
        raise TypeError(f'$binary base64 must be a string: {doc}')
    if not isinstance(subtype, str) or len(subtype) > 2:
        raise TypeError(f'$binary subType must be a string at most 2 characters: {doc}')
    if len(binary) != 2:
        raise TypeError(f'$binary must include only "base64" and "subType" components: {doc}')
    data = base64.b64decode(b64.encode())
    return _binary_or_uuid(data, int(subtype, 16), json_options)