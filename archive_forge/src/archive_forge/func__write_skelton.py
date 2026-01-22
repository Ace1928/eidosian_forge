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
def _write_skelton(self, file: BinaryIO):
    file.seek(0, 0)
    write_bytes(file, MAGIC_7Z)
    write_byte(file, self.version[0])
    write_byte(file, self.version[1])
    write_uint32(file, 1)
    write_real_uint64(file, 2)
    write_real_uint64(file, 3)
    write_uint32(file, 4)