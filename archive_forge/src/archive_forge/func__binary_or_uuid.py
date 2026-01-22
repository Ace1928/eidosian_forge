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
def _binary_or_uuid(data: Any, subtype: int, json_options: JSONOptions) -> Union[Binary, uuid.UUID]:
    if subtype in ALL_UUID_SUBTYPES:
        uuid_representation = json_options.uuid_representation
        binary_value = Binary(data, subtype)
        if uuid_representation == UuidRepresentation.UNSPECIFIED:
            return binary_value
        if subtype == UUID_SUBTYPE:
            uuid_representation = UuidRepresentation.STANDARD
        elif uuid_representation == UuidRepresentation.STANDARD:
            uuid_representation = UuidRepresentation.PYTHON_LEGACY
        return binary_value.as_uuid(uuid_representation)
    if subtype == 0:
        return cast(uuid.UUID, data)
    return Binary(data, subtype)