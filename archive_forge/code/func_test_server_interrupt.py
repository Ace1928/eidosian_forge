import os
import queue
import socket
import tempfile
import threading
import types
import uuid
import urllib.parse  # noqa: WPS301
import pytest
import requests
import requests_unixsocket
from pypytools.gc.custom import DefaultGc
from .._compat import bton, ntob
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS, SYS_PLATFORM
from ..server import IS_UID_GID_RESOLVABLE, Gateway, HTTPServer
from ..workers.threadpool import ThreadPool
from ..testing import (
@pytest.mark.parametrize('exc_cls', (IOError, KeyboardInterrupt, OSError, RuntimeError))
def test_server_interrupt(exc_cls):
    """Check that assigning interrupt stops the server."""
    interrupt_msg = 'should catch {uuid!s}'.format(uuid=uuid.uuid4())
    raise_marker_sentinel = object()
    httpserver = HTTPServer(bind_addr=(ANY_INTERFACE_IPV4, EPHEMERAL_PORT), gateway=Gateway)
    result_q = queue.Queue()

    def serve_thread():
        try:
            httpserver.serve()
        except exc_cls as e:
            if str(e) == interrupt_msg:
                result_q.put(raise_marker_sentinel)
    httpserver.prepare()
    serve_thread = threading.Thread(target=serve_thread)
    serve_thread.start()
    serve_thread.join(0.5)
    assert serve_thread.is_alive()
    httpserver.interrupt = exc_cls(interrupt_msg)
    serve_thread.join(0.5)
    assert not serve_thread.is_alive()
    assert result_q.get_nowait() is raise_marker_sentinel