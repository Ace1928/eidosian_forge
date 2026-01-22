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
def _op_msg_no_header(flags: int, command: Mapping[str, Any], identifier: str, docs: Optional[list[Mapping[str, Any]]], opts: CodecOptions) -> tuple[bytes, int, int]:
    """Get a OP_MSG message.

    Note: this method handles multiple documents in a type one payload but
    it does not perform batch splitting and the total message size is
    only checked *after* generating the entire message.
    """
    encoded = _dict_to_bson(command, False, opts)
    flags_type = _pack_op_msg_flags_type(flags, 0)
    total_size = len(encoded)
    max_doc_size = 0
    if identifier and docs is not None:
        type_one = _pack_byte(1)
        cstring = _make_c_string(identifier)
        encoded_docs = [_dict_to_bson(doc, False, opts) for doc in docs]
        size = len(cstring) + sum((len(doc) for doc in encoded_docs)) + 4
        encoded_size = _pack_int(size)
        total_size += size
        max_doc_size = max((len(doc) for doc in encoded_docs))
        data = [flags_type, encoded, type_one, encoded_size, cstring, *encoded_docs]
    else:
        data = [flags_type, encoded]
    return (b''.join(data), total_size, max_doc_size)