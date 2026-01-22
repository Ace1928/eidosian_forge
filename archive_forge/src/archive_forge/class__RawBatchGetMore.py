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
class _RawBatchGetMore(_GetMore):

    def use_command(self, conn: Connection) -> bool:
        super().use_command(conn)
        if conn.max_wire_version >= 8:
            return True
        elif not self.exhaust:
            return True
        return False