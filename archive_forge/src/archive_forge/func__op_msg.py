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
def _op_msg(flags: int, command: MutableMapping[str, Any], dbname: str, read_preference: Optional[_ServerMode], opts: CodecOptions, ctx: Union[SnappyContext, ZlibContext, ZstdContext, None]=None) -> tuple[int, bytes, int, int]:
    """Get a OP_MSG message."""
    command['$db'] = dbname
    if read_preference is not None and '$readPreference' not in command:
        if read_preference.mode:
            command['$readPreference'] = read_preference.document
    name = next(iter(command))
    try:
        identifier = _FIELD_MAP[name]
        docs = command.pop(identifier)
    except KeyError:
        identifier = ''
        docs = None
    try:
        if ctx:
            return _op_msg_compressed(flags, command, identifier, docs, opts, ctx)
        return _op_msg_uncompressed(flags, command, identifier, docs, opts)
    finally:
        if identifier:
            command[identifier] = docs