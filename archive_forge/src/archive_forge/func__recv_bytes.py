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
def _recv_bytes(self, maxsize=None):
    if self._got_empty_message:
        self._got_empty_message = False
        return io.BytesIO()
    else:
        bsize = 128 if maxsize is None else min(maxsize, 128)
        try:
            ov, err = _winapi.ReadFile(self._handle, bsize, overlapped=True)
            try:
                if err == _winapi.ERROR_IO_PENDING:
                    waitres = _winapi.WaitForMultipleObjects([ov.event], False, INFINITE)
                    assert waitres == WAIT_OBJECT_0
            except:
                ov.cancel()
                raise
            finally:
                nread, err = ov.GetOverlappedResult(True)
                if err == 0:
                    f = io.BytesIO()
                    f.write(ov.getbuffer())
                    return f
                elif err == _winapi.ERROR_MORE_DATA:
                    return self._get_more_data(ov, maxsize)
        except OSError as e:
            if e.winerror == _winapi.ERROR_BROKEN_PIPE:
                raise EOFError
            else:
                raise
    raise RuntimeError("shouldn't get here; expected KeyboardInterrupt")