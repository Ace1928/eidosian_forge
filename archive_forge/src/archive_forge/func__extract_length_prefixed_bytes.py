import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _extract_length_prefixed_bytes(self):
    if self._in_buffer_len < 4:
        raise _NeedMoreBytes(4)
    length, = struct.unpack('!L', self._get_in_bytes(4))
    end_of_bytes = 4 + length
    if self._in_buffer_len < end_of_bytes:
        raise _NeedMoreBytes(end_of_bytes)
    in_buf = self._get_in_buffer()
    bytes = in_buf[4:end_of_bytes]
    self._set_in_buffer(in_buf[end_of_bytes:])
    return bytes