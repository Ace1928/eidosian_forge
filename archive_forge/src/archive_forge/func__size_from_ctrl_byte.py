import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _size_from_ctrl_byte(self, ctrl_byte: int, offset: int, type_num: int) -> Tuple[int, int]:
    size = ctrl_byte & 31
    if type_num == 1 or size < 29:
        return (size, offset)
    if size == 29:
        size = 29 + self._buffer[offset]
        return (size, offset + 1)
    if size == 30:
        new_offset = offset + 2
        size_bytes = self._buffer[offset:new_offset]
        size = 285 + struct.unpack(b'!H', size_bytes)[0]
        return (size, new_offset)
    new_offset = offset + 3
    size_bytes = self._buffer[offset:new_offset]
    size = struct.unpack(b'!I', b'\x00' + size_bytes)[0] + 65821
    return (size, new_offset)