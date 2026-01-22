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
def _retrieve_coders_info(self, file: BinaryIO):
    pid = file.read(1)
    if pid != PROPERTY.CODERS_UNPACK_SIZE:
        raise Bad7zFile('coders unpack size id expected but %s found' % repr(pid))
    for folder in self.folders:
        for c in folder.coders:
            for _ in range(c['numoutstreams']):
                folder.unpacksizes.append(read_uint64(file))
    pid = file.read(1)
    if pid == PROPERTY.CRC:
        defined = read_boolean(file, self.numfolders, checkall=True)
        crcs = read_crcs(file, self.numfolders)
        for idx, folder in enumerate(self.folders):
            folder.digestdefined = defined[idx]
            folder.crc = crcs[idx]
        pid = file.read(1)
    if pid != PROPERTY.END:
        raise Bad7zFile('end id expected but 0x{:02x} found at 0x{:08x}'.format(ord(pid), file.tell()))