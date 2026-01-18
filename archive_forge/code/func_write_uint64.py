import functools
import io
import operator
import os
import struct
from binascii import unhexlify
from functools import reduce
from io import BytesIO
from operator import and_, or_
from struct import pack, unpack
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from py7zr.compressor import SevenZipCompressor, SevenZipDecompressor
from py7zr.exceptions import Bad7zFile
from py7zr.helpers import ArchiveTimestamp, calculate_crc32
from py7zr.properties import DEFAULT_FILTERS, MAGIC_7Z, PROPERTY
def write_uint64(file: Union[BinaryIO, WriteWithCrc], value: int):
    """
    UINT64 means real UINT64 encoded with the following scheme:

    |  Size of encoding sequence depends from first byte:
    |  First_Byte  Extra_Bytes        Value
    |  (binary)
    |  0xxxxxxx               : ( xxxxxxx           )
    |  10xxxxxx    BYTE y[1]  : (  xxxxxx << (8 * 1)) + y
    |  110xxxxx    BYTE y[2]  : (   xxxxx << (8 * 2)) + y
    |  ...
    |  1111110x    BYTE y[6]  : (       x << (8 * 6)) + y
    |  11111110    BYTE y[7]  :                         y
    |  11111111    BYTE y[8]  :                         y
    """
    if value < 128:
        file.write(pack('B', value))
        return
    if value > 144115188075855871:
        file.write(b'\xff')
        file.write(value.to_bytes(8, 'little'))
        return
    byte_length = (value.bit_length() + 7) // 8
    ba = bytearray(value.to_bytes(byte_length, 'little'))
    high_byte = int(ba[-1])
    if high_byte < 2 << 8 - byte_length - 1:
        for x in range(byte_length - 1):
            high_byte |= 128 >> x
        file.write(pack('B', high_byte))
        file.write(ba[:byte_length - 1])
    else:
        mask = 128
        for x in range(byte_length):
            mask |= 128 >> x
        file.write(pack('B', mask))
        file.write(ba)