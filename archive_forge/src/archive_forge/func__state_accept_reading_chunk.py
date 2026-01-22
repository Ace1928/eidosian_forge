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
def _state_accept_reading_chunk(self):
    in_buf = self._get_in_buffer()
    in_buffer_len = len(in_buf)
    self.chunk_in_progress += in_buf[:self.bytes_left]
    self._set_in_buffer(in_buf[self.bytes_left:])
    self.bytes_left -= in_buffer_len
    if self.bytes_left <= 0:
        self.bytes_left = None
        if self.error:
            self.error_in_progress.append(self.chunk_in_progress)
        else:
            self.chunks.append(self.chunk_in_progress)
        self.chunk_in_progress = None
        self.state_accept = self._state_accept_expecting_length