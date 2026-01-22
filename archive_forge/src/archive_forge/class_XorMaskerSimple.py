import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
class XorMaskerSimple:

    def __init__(self, masking_key: bytes) -> None:
        self._masking_key = masking_key

    def process(self, data: bytes) -> bytes:
        if data:
            data_array = bytearray(data)
            a, b, c, d = (_XOR_TABLE[n] for n in self._masking_key)
            data_array[::4] = data_array[::4].translate(a)
            data_array[1::4] = data_array[1::4].translate(b)
            data_array[2::4] = data_array[2::4].translate(c)
            data_array[3::4] = data_array[3::4].translate(d)
            key_rotation = len(data) % 4
            self._masking_key = self._masking_key[key_rotation:] + self._masking_key[:key_rotation]
            return bytes(data_array)
        return data