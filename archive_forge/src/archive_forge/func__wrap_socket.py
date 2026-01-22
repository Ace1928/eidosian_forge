from eventlet.patcher import slurp_properties
import sys
from eventlet import greenio, hubs
from eventlet.greenio import (
from eventlet.hubs import trampoline, IOClosed
from eventlet.support import get_errno, PY33
from contextlib import contextmanager
@staticmethod
def _wrap_socket(sock, keyfile, certfile, server_side, cert_reqs, ssl_version, ca_certs, do_handshake_on_connect, ciphers):
    context = _original_sslcontext(protocol=ssl_version)
    context.options |= cert_reqs
    if certfile or keyfile:
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if ca_certs:
        context.load_verify_locations(ca_certs)
    if ciphers:
        context.set_ciphers(ciphers)
    return context.wrap_socket(sock=sock, server_side=server_side, do_handshake_on_connect=do_handshake_on_connect)