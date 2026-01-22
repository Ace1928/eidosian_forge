import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
def _ssl_io_loop(self, func, *args):
    """Performs an I/O loop between incoming/outgoing and the socket."""
    should_loop = True
    ret = None
    while should_loop:
        errno = None
        try:
            ret = func(*args)
        except ssl.SSLError as e:
            if e.errno not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE):
                raise e
            errno = e.errno
        buf = self.outgoing.read()
        self.socket.sendall(buf)
        if errno is None:
            should_loop = False
        elif errno == ssl.SSL_ERROR_WANT_READ:
            buf = self.socket.recv(SSL_BLOCKSIZE)
            if buf:
                self.incoming.write(buf)
            else:
                self.incoming.write_eof()
    return ret