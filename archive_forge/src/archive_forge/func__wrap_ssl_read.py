import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
def _wrap_ssl_read(self, len, buffer=None):
    try:
        return self._ssl_io_loop(self.sslobj.read, len, buffer)
    except ssl.SSLError as e:
        if e.errno == ssl.SSL_ERROR_EOF and self.suppress_ragged_eofs:
            return 0
        else:
            raise