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
def _query_uncompressed(options: int, collection_name: str, num_to_skip: int, num_to_return: int, query: Mapping[str, Any], field_selector: Optional[Mapping[str, Any]], opts: CodecOptions) -> tuple[int, bytes, int]:
    """Internal query message helper."""
    op_query, max_bson_size = _query_impl(options, collection_name, num_to_skip, num_to_return, query, field_selector, opts)
    rid, msg = __pack_message(2004, op_query)
    return (rid, msg, max_bson_size)