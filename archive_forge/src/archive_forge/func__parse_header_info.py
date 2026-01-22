from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
def _parse_header_info(self):
    """If this stream contains object data, parse the header info and skip the
        stream to a point where each read will yield object content

        :return: parsed type_string, size"""
    maxb = 8192
    self._s = maxb
    hdr = self.read(maxb)
    hdrend = hdr.find(NULL_BYTE)
    typ, size = hdr[:hdrend].split(BYTE_SPACE)
    size = int(size)
    self._s = size
    self._br = 0
    hdrend += 1
    self._buf = BytesIO(hdr[hdrend:])
    self._buflen = len(hdr) - hdrend
    self._phi = True
    return (typ, size)