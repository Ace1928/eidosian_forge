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
class PipeConnection(_ConnectionBase):
    """
        Connection class based on a Windows named pipe.
        Overlapped I/O is used, so the handles must have been created
        with FILE_FLAG_OVERLAPPED.
        """
    _got_empty_message = False

    def _close(self, _CloseHandle=_winapi.CloseHandle):
        _CloseHandle(self._handle)

    def _send_bytes(self, buf):
        ov, err = _winapi.WriteFile(self._handle, buf, overlapped=True)
        try:
            if err == _winapi.ERROR_IO_PENDING:
                waitres = _winapi.WaitForMultipleObjects([ov.event], False, INFINITE)
                assert waitres == WAIT_OBJECT_0
        except:
            ov.cancel()
            raise
        finally:
            nwritten, err = ov.GetOverlappedResult(True)
        assert err == 0
        assert nwritten == len(buf)

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

    def _poll(self, timeout):
        if self._got_empty_message or _winapi.PeekNamedPipe(self._handle)[0] != 0:
            return True
        return bool(wait([self], timeout))

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