import array
import ctypes.wintypes
import platform
import struct
from paramiko.common import zero_byte
from paramiko.util import b
import _thread as thread
from . import _winapi
class PageantConnection:
    """
    Mock "connection" to an agent which roughly approximates the behavior of
    a unix local-domain socket (as used by Agent).  Requests are sent to the
    pageant daemon via special Windows magick, and responses are buffered back
    for subsequent reads.
    """

    def __init__(self):
        self._response = None

    def send(self, data):
        self._response = _query_pageant(data)

    def recv(self, n):
        if self._response is None:
            return ''
        ret = self._response[:n]
        self._response = self._response[n:]
        if self._response == '':
            self._response = None
        return ret

    def close(self):
        pass