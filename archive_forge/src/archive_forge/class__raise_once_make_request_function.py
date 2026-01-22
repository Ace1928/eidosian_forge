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
class _raise_once_make_request_function(object):
    """Callable that can mimic `_make_request()`.

            Raises the given exception on its first call, but returns a
            successful response on subsequent calls.
            """

    def __init__(self, ex):
        super(_raise_once_make_request_function, self).__init__()
        self._ex = ex

    def __call__(self, *args, **kwargs):
        if self._ex:
            ex, self._ex = (self._ex, None)
            raise ex()
        response = httplib.HTTPResponse(MockSock)
        response.fp = MockChunkedEncodingResponse([b'f', b'o', b'o'])
        response.headers = response.msg = HTTPHeaderDict()
        return response