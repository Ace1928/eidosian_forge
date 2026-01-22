import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _loopback_for_cert(certificate, private_key, certificate_chain):
    """Create a loopback connection to parse a cert with a private key."""
    context = ssl.create_default_context(cafile=certificate_chain)
    context.load_cert_chain(certificate, private_key)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    client, server = socket.socketpair()
    try:
        thread = threading.Thread(target=_loopback_for_cert_thread, args=(context, server))
        try:
            thread.start()
            with context.wrap_socket(client, do_handshake_on_connect=True, server_side=False) as ssl_sock:
                ssl_sock.recv(4)
                return ssl_sock.getpeercert()
        finally:
            thread.join()
    finally:
        client.close()
        server.close()