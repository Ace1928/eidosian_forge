import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
def _read_all_headers(self):
    pos = 0
    msgs = []
    buffer_len = len(self._buffer)
    while pos < buffer_len:
        header = self._read_header(pos)
        msgs.append((header, pos))
        pos += self.LOG_OVERHEAD + header[1]
    return msgs