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
def _state_accept_reading_body(self):
    in_buf = self._get_in_buffer()
    self._body += in_buf
    self.bytes_left -= len(in_buf)
    self._set_in_buffer(None)
    if self.bytes_left <= 0:
        if self.bytes_left != 0:
            self._trailer_buffer = self._body[self.bytes_left:]
            self._body = self._body[:self.bytes_left]
        self.bytes_left = None
        self.state_accept = self._state_accept_reading_trailer