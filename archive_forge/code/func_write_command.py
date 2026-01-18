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
@_handle_reauth
def write_command(self, cmd: MutableMapping[str, Any], request_id: int, msg: bytes, docs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """A proxy for SocketInfo.write_command that handles event publishing."""
    if self.publish:
        assert self.start_time is not None
        duration = datetime.datetime.now() - self.start_time
        self._start(cmd, request_id, docs)
        start = datetime.datetime.now()
    try:
        reply = self.conn.write_command(request_id, msg, self.codec)
        if self.publish:
            duration = datetime.datetime.now() - start + duration
            self._succeed(request_id, reply, duration)
    except Exception as exc:
        if self.publish:
            duration = datetime.datetime.now() - start + duration
            if isinstance(exc, (NotPrimaryError, OperationFailure)):
                failure: _DocumentOut = exc.details
            else:
                failure = _convert_exception(exc)
            self._fail(request_id, failure, duration)
        raise
    finally:
        self.start_time = datetime.datetime.now()
    return reply