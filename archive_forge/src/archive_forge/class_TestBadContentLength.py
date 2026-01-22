from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
class TestBadContentLength(SocketDummyServerTestCase):

    def test_enforce_content_length_get(self):
        done_event = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: 22\r\nContent-type: text/plain\r\n\r\nhello, world')
            done_event.wait(LONG_TIMEOUT)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as conn:
            get_response = conn.request('GET', url='/', preload_content=False, enforce_content_length=True)
            data = get_response.stream(100)
            next(data)
            try:
                next(data)
                assert False
            except ProtocolError as e:
                assert '12 bytes read, 10 more expected' in str(e)
            done_event.set()

    def test_enforce_content_length_no_body(self):
        done_event = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: 22\r\nContent-type: text/plain\r\n\r\n')
            done_event.wait(1)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as conn:
            head_response = conn.request('HEAD', url='/', preload_content=False, enforce_content_length=True)
            data = [chunk for chunk in head_response.stream(1)]
            assert len(data) == 0
            done_event.set()