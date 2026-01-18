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
def test_response_headers_are_returned_in_the_original_order(self):
    K = 16
    expected_response_headers = [('X-Header-%d' % i, str(i)) for i in reversed(range(K))]

    def socket_handler(listener):
        sock = listener.accept()[0]
        buf = b''
        while not buf.endswith(b'\r\n\r\n'):
            buf += sock.recv(65536)
        sock.send(b'HTTP/1.1 200 OK\r\n' + b'\r\n'.join([k.encode('utf8') + b': ' + v.encode('utf8') for k, v in expected_response_headers]) + b'\r\n')
        sock.close()
    self._start_server(socket_handler)
    with HTTPConnectionPool(self.host, self.port) as pool:
        r = pool.request('GET', '/', retries=0)
        actual_response_headers = [(k, v) for k, v in r.headers.items() if k.startswith('X-Header-')]
        assert expected_response_headers == actual_response_headers