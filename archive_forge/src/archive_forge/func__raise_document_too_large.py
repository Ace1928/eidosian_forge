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
def _raise_document_too_large(operation: str, doc_size: int, max_size: int) -> NoReturn:
    """Internal helper for raising DocumentTooLarge."""
    if operation == 'insert':
        raise DocumentTooLarge('BSON document too large (%d bytes) - the connected server supports BSON document sizes up to %d bytes.' % (doc_size, max_size))
    else:
        raise DocumentTooLarge(f'{operation!r} command document too large')