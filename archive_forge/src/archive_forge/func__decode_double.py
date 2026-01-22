import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _decode_double(self, size: int, offset: int) -> Tuple[float, int]:
    self._verify_size(size, 8)
    new_offset = offset + size
    packed_bytes = self._buffer[offset:new_offset]
    value, = struct.unpack(b'!d', packed_bytes)
    return (value, new_offset)