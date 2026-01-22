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
class _CursorAddress(tuple):
    """The server address (host, port) of a cursor, with namespace property."""
    __namespace: Any

    def __new__(cls, address: _Address, namespace: str) -> _CursorAddress:
        self = tuple.__new__(cls, address)
        self.__namespace = namespace
        return self

    @property
    def namespace(self) -> str:
        """The namespace this cursor."""
        return self.__namespace

    def __hash__(self) -> int:
        return (*self, self.__namespace).__hash__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CursorAddress):
            return tuple(self) == tuple(other) and self.namespace == other.namespace
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        return not self == other