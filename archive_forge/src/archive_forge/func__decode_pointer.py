import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _decode_pointer(self, size: int, offset: int) -> Tuple[Record, int]:
    pointer_size = (size >> 3) + 1
    buf = self._buffer[offset:offset + pointer_size]
    new_offset = offset + pointer_size
    if pointer_size == 1:
        buf = bytes([size & 7]) + buf
        pointer = struct.unpack(b'!H', buf)[0] + self._pointer_base
    elif pointer_size == 2:
        buf = b'\x00' + bytes([size & 7]) + buf
        pointer = struct.unpack(b'!I', buf)[0] + 2048 + self._pointer_base
    elif pointer_size == 3:
        buf = bytes([size & 7]) + buf
        pointer = struct.unpack(b'!I', buf)[0] + 526336 + self._pointer_base
    else:
        pointer = struct.unpack(b'!I', buf)[0] + self._pointer_base
    if self._pointer_test:
        return (pointer, new_offset)
    value, _ = self.decode(pointer)
    return (value, new_offset)