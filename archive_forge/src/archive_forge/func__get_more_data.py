import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _get_more_data(self, ov, maxsize):
    buf = ov.getbuffer()
    f = io.BytesIO()
    f.write(buf)
    left = _winapi.PeekNamedPipe(self._handle)[1]
    assert left > 0
    if maxsize is not None and len(buf) + left > maxsize:
        self._bad_message_length()
    ov, err = _winapi.ReadFile(self._handle, left, overlapped=True)
    rbytes, err = ov.GetOverlappedResult(True)
    assert err == 0
    assert rbytes == left
    f.write(ov.getbuffer())
    return f