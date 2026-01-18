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
def test_pool_size(self):
    POOL_SIZE = 1
    with HTTPConnectionPool(host='localhost', maxsize=POOL_SIZE, block=True) as pool:

        def _raise(ex):
            raise ex()

        def _test(exception, expect, reason=None):
            pool._make_request = lambda *args, **kwargs: _raise(exception)
            with pytest.raises(expect) as excinfo:
                pool.request('GET', '/')
            if reason is not None:
                assert isinstance(excinfo.value.reason, reason)
            assert pool.pool.qsize() == POOL_SIZE
        _test(BaseSSLError, MaxRetryError, SSLError)
        _test(CertificateError, MaxRetryError, SSLError)
        pool._make_request = lambda *args, **kwargs: _raise(HTTPException)
        with pytest.raises(MaxRetryError):
            pool.request('GET', '/', retries=1, pool_timeout=SHORT_TIMEOUT)
        assert pool.pool.qsize() == POOL_SIZE