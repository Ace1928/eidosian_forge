from eventlet.patcher import slurp_properties
import sys
from eventlet import greenio, hubs
from eventlet.greenio import (
from eventlet.hubs import trampoline, IOClosed
from eventlet.support import get_errno, PY33
from contextlib import contextmanager
@contextmanager
def _original_ssl_context(*args, **kwargs):
    tmp_sslcontext = _original_wrap_socket.__globals__.get('SSLContext', None)
    tmp_sslsocket = _original_sslsocket._create.__globals__.get('SSLSocket', None)
    _original_sslsocket._create.__globals__['SSLSocket'] = _original_sslsocket
    _original_wrap_socket.__globals__['SSLContext'] = _original_sslcontext
    try:
        yield
    finally:
        _original_wrap_socket.__globals__['SSLContext'] = tmp_sslcontext
        _original_sslsocket._create.__globals__['SSLSocket'] = tmp_sslsocket