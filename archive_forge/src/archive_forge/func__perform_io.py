import socket
import ssl
import sys
import typing
from functools import partial
from .._exceptions import (
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, NetworkBackend, NetworkStream
def _perform_io(self, func: typing.Callable[..., typing.Any]) -> typing.Any:
    ret = None
    while True:
        errno = None
        try:
            ret = func()
        except (ssl.SSLWantReadError, ssl.SSLWantWriteError) as e:
            errno = e.errno
        self._sock.sendall(self._outgoing.read())
        if errno == ssl.SSL_ERROR_WANT_READ:
            buf = self._sock.recv(self.TLS_RECORD_SIZE)
            if buf:
                self._incoming.write(buf)
            else:
                self._incoming.write_eof()
        if errno is None:
            return ret