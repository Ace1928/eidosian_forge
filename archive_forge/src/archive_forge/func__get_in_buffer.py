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
def _get_in_buffer(self):
    if len(self._in_buffer_list) == 1:
        return self._in_buffer_list[0]
    in_buffer = b''.join(self._in_buffer_list)
    if len(in_buffer) != self._in_buffer_len:
        raise AssertionError('Length of buffer did not match expected value: %s != %s' % self._in_buffer_len, len(in_buffer))
    self._in_buffer_list = [in_buffer]
    return in_buffer