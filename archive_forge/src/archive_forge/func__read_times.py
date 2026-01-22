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
def _read_times(self, fp: BinaryIO, name: str) -> None:
    defined = read_boolean(fp, len(self.files), checkall=True)
    external = fp.read(1)
    assert external == b'\x00'
    for i, f in enumerate(self.files):
        f[name] = ArchiveTimestamp(read_real_uint64(fp)[0]) if defined[i] else None