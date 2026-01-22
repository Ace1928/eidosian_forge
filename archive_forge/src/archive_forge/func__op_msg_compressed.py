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
def _op_msg_compressed(flags: int, command: Mapping[str, Any], identifier: str, docs: Optional[list[Mapping[str, Any]]], opts: CodecOptions, ctx: Union[SnappyContext, ZlibContext, ZstdContext]) -> tuple[int, bytes, int, int]:
    """Internal OP_MSG message helper."""
    msg, total_size, max_bson_size = _op_msg_no_header(flags, command, identifier, docs, opts)
    rid, msg = _compress(2013, msg, ctx)
    return (rid, msg, total_size, max_bson_size)