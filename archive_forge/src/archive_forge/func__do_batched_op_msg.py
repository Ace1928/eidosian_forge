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
def _do_batched_op_msg(namespace: str, operation: int, command: MutableMapping[str, Any], docs: list[Mapping[str, Any]], opts: CodecOptions, ctx: _BulkWriteContext) -> tuple[int, bytes, list[Mapping[str, Any]]]:
    """Create the next batched insert, update, or delete operation
    using OP_MSG.
    """
    command['$db'] = namespace.split('.', 1)[0]
    if 'writeConcern' in command:
        ack = bool(command['writeConcern'].get('w', 1))
    else:
        ack = True
    if ctx.conn.compression_context:
        return _batched_op_msg_compressed(operation, command, docs, ack, opts, ctx)
    return _batched_op_msg(operation, command, docs, ack, opts, ctx)