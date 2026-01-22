import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _loopback_for_cert_thread(context, server):
    """Wrap a socket in ssl and perform the server-side handshake."""
    with suppress(ssl.SSLError, OSError):
        with context.wrap_socket(server, do_handshake_on_connect=True, server_side=True) as ssl_sock:
            ssl_sock.send(b'0000')