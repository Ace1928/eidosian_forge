import socket
import sys
import threading
import time
from . import Adapter
from .. import errors, server as cheroot_server
from ..makefile import StreamReader, StreamWriter
class SSLFileobjectMixin:
    """Base mixin for a TLS socket stream."""
    ssl_timeout = 3
    ssl_retry = 0.01

    def _safe_call(self, is_reader, call, *args, **kwargs):
        """Wrap the given call with TLS error-trapping.

        is_reader: if False EOF errors will be raised. If True, EOF errors
        will return "" (to emulate normal sockets).
        """
        start = time.time()
        while True:
            try:
                return call(*args, **kwargs)
            except SSL.WantReadError:
                time.sleep(self.ssl_retry)
            except SSL.WantWriteError:
                time.sleep(self.ssl_retry)
            except SSL.SysCallError as e:
                if is_reader and e.args == (-1, 'Unexpected EOF'):
                    return b''
                errnum = e.args[0]
                if is_reader and errnum in errors.socket_errors_to_ignore:
                    return b''
                raise socket.error(errnum)
            except SSL.Error as e:
                if is_reader and e.args == (-1, 'Unexpected EOF'):
                    return b''
                thirdarg = None
                try:
                    thirdarg = e.args[0][0][2]
                except IndexError:
                    pass
                if thirdarg == 'http request':
                    raise errors.NoSSLError()
                raise errors.FatalSSLAlert(*e.args)
            if time.time() - start > self.ssl_timeout:
                raise socket.timeout('timed out')

    def recv(self, size):
        """Receive message of a size from the socket."""
        return self._safe_call(True, super(SSLFileobjectMixin, self).recv, size)

    def readline(self, size=-1):
        """Receive message of a size from the socket.

        Matches the following interface:
        https://docs.python.org/3/library/io.html#io.IOBase.readline
        """
        return self._safe_call(True, super(SSLFileobjectMixin, self).readline, size)

    def sendall(self, *args, **kwargs):
        """Send whole message to the socket."""
        return self._safe_call(False, super(SSLFileobjectMixin, self).sendall, *args, **kwargs)

    def send(self, *args, **kwargs):
        """Send some part of message to the socket."""
        return self._safe_call(False, super(SSLFileobjectMixin, self).send, *args, **kwargs)