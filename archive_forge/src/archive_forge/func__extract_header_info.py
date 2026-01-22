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
def _extract_header_info(self, fp: BinaryIO) -> None:
    pid = fp.read(1)
    if pid == PROPERTY.MAIN_STREAMS_INFO:
        self.main_streams = StreamsInfo.retrieve(fp)
        pid = fp.read(1)
    if pid == PROPERTY.FILES_INFO:
        self.files_info = FilesInfo.retrieve(fp)
        pid = fp.read(1)
    if pid != PROPERTY.END:
        raise Bad7zFile('end id expected but %s found' % repr(pid))