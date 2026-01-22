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
class TestCookies(SocketDummyServerTestCase):

    def test_multi_setcookie(self):

        def multicookie_response_handler(listener):
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nSet-Cookie: foo=1\r\nSet-Cookie: bar=1\r\n\r\n')
            sock.close()
        self._start_server(multicookie_response_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/', retries=0)
            assert r.headers == {'set-cookie': 'foo=1, bar=1'}
            assert r.headers.getlist('set-cookie') == ['foo=1', 'bar=1']