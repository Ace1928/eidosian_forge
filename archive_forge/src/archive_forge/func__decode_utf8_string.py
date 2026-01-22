import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _decode_utf8_string(self, size: int, offset: int) -> Tuple[str, int]:
    new_offset = offset + size
    return (self._buffer[offset:new_offset].decode('utf-8'), new_offset)