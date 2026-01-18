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
@pytest.mark.parametrize('a, b', [('https://google.com/', 'http://google.com/'), ('http://google.com/', 'https://google.com/'), ('http://yahoo.com/', 'http://google.com/'), ('http://google.com:42', 'https://google.com/abracadabra'), ('http://google.com', 'https://google.net/'), ('http://google.com:42', 'http://google.com'), ('https://google.com:42', 'https://google.com'), ('http://google.com:443', 'http://google.com'), ('https://google.com:80', 'https://google.com'), ('http://google.com:443', 'https://google.com'), ('https://google.com:80', 'http://google.com'), ('https://google.com:443', 'http://google.com'), ('http://google.com:80', 'https://google.com'), ('http://[dead::beef]', 'https://[dead::beef%en5]/')])
def test_not_same_host(self, a, b):
    with connection_from_url(a) as c:
        assert not c.is_same_host(b)
    with connection_from_url(b) as c:
        assert not c.is_same_host(a)