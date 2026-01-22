from __future__ import annotations
import datetime
import random
import struct
from io import BytesIO as _BytesIO
from typing import (
import bson
from bson import CodecOptions, _decode_selective, _dict_to_bson, _make_c_string, encode
from bson.int64 import Int64
from bson.raw_bson import (
from bson.son import SON
from pymongo.errors import (
from pymongo.hello import HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def _get_more_impl(collection_name: str, num_to_return: int, cursor_id: int) -> bytes:
    """Get an OP_GET_MORE message."""
    return b''.join([_ZERO_32, _make_c_string(collection_name), _pack_int(num_to_return), _pack_long_long(cursor_id)])