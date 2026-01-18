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
@pytest.mark.skipif(not hasattr(socket, 'SO_REUSEPORT'), reason='socket.SO_REUSEPORT is not supported on this platform')
@pytest.mark.parametrize('ip_addr', (ANY_INTERFACE_IPV4, ANY_INTERFACE_IPV6))
def test_reuse_port(http_server, ip_addr, mocker):
    """Check that port initialized externally can be reused."""
    family = socket.getaddrinfo(ip_addr, EPHEMERAL_PORT)[0][0]
    s = socket.socket(family)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind((ip_addr, EPHEMERAL_PORT))
    server = HTTPServer(bind_addr=s.getsockname()[:2], gateway=Gateway, reuse_port=True)
    spy = mocker.spy(server, 'prepare')
    server.prepare()
    server.stop()
    s.close()
    assert spy.spy_exception is None