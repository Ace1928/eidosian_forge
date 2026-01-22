import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def _make_chunk(self):
    raw_chunk = self._raw.read(self._chunk_size)
    hex_len = hex(len(raw_chunk))[2:].encode('ascii')
    self._complete = not raw_chunk
    if self._checksum:
        self._checksum.update(raw_chunk)
    if self._checksum and self._complete:
        name = self._checksum_name.encode('ascii')
        checksum = self._checksum.b64digest().encode('ascii')
        return b'0\r\n%s:%s\r\n\r\n' % (name, checksum)
    return b'%s\r\n%s\r\n' % (hex_len, raw_chunk)