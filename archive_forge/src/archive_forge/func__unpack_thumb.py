import struct
from typing import Union
def _unpack_thumb(self, b: Union[bytearray, bytes, memoryview]) -> int:
    return (b[1] & 7) << 19 | b[0] << 11 | (b[3] & 7) << 8 | b[2]