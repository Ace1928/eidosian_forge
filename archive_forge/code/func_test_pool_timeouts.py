from __future__ import absolute_import
import ssl
from socket import error as SocketError
from ssl import SSLError as BaseSSLError
from test import SHORT_TIMEOUT
import pytest
from mock import Mock
from dummyserver.server import DEFAULT_CA
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import (
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.packages.six.moves.http_client import HTTPException
from urllib3.packages.six.moves.queue import Empty
from urllib3.response import HTTPResponse
from urllib3.util.ssl_match_hostname import CertificateError
from urllib3.util.timeout import Timeout
from .test_response import MockChunkedEncodingResponse, MockSock
def test_pool_timeouts(self):
    with HTTPConnectionPool(host='localhost') as pool:
        conn = pool._new_conn()
        assert conn.__class__ == HTTPConnection
        assert pool.timeout.__class__ == Timeout
        assert pool.timeout._read == Timeout.DEFAULT_TIMEOUT
        assert pool.timeout._connect == Timeout.DEFAULT_TIMEOUT
        assert pool.timeout.total is None
        pool = HTTPConnectionPool(host='localhost', timeout=SHORT_TIMEOUT)
        assert pool.timeout._read == SHORT_TIMEOUT
        assert pool.timeout._connect == SHORT_TIMEOUT
        assert pool.timeout.total is None