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
def unpack_response(self, cursor_id: Optional[int]=None, codec_options: CodecOptions=_UNICODE_REPLACE_CODEC_OPTIONS, user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> list[dict[str, Any]]:
    """Unpack a OP_MSG command response.

        :Parameters:
          - `cursor_id` (optional): Ignored, for compatibility with _OpReply.
          - `codec_options` (optional): an instance of
            :class:`~bson.codec_options.CodecOptions`
          - `user_fields` (optional): Response fields that should be decoded
            using the TypeDecoders from codec_options, passed to
            bson._decode_all_selective.
        """
    assert not legacy_response
    return bson._decode_all_selective(self.payload_document, codec_options, user_fields)