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
def _state_accept_expecting_protocol_version(self):
    needed_bytes = len(MESSAGE_VERSION_THREE) - self._in_buffer_len
    in_buf = self._get_in_buffer()
    if needed_bytes > 0:
        if not MESSAGE_VERSION_THREE.startswith(in_buf):
            raise errors.UnexpectedProtocolVersionMarker(in_buf)
        raise _NeedMoreBytes(len(MESSAGE_VERSION_THREE))
    if not in_buf.startswith(MESSAGE_VERSION_THREE):
        raise errors.UnexpectedProtocolVersionMarker(in_buf)
    self._set_in_buffer(in_buf[len(MESSAGE_VERSION_THREE):])
    self.state_accept = self._state_accept_expecting_headers