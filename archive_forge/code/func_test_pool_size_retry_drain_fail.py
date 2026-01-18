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
def test_pool_size_retry_drain_fail(self):

    def socket_handler(listener):
        for _ in range(2):
            sock = listener.accept()[0]
            while not sock.recv(65536).endswith(b'\r\n\r\n'):
                pass
            sock.send(b'HTTP/1.1 404 NOT FOUND\r\nContent-Length: 1000\r\nContent-Type: text/plain\r\n\r\n')
            sock.close()
    self._start_server(socket_handler)
    retries = Retry(total=1, raise_on_status=False, status_forcelist=[404])
    with HTTPConnectionPool(self.host, self.port, maxsize=10, retries=retries, block=True) as pool:
        pool.urlopen('GET', '/not_found', preload_content=False)
        assert pool.num_connections == 1