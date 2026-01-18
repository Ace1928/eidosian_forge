import io
import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
from test import LONG_TIMEOUT, SHORT_TIMEOUT, onlyPy2
from threading import Event
import mock
import pytest
import six
from dummyserver.server import HAS_IPV6_AND_DNS, NoIPv6Warning
from dummyserver.testcase import HTTPDummyServerTestCase, SocketDummyServerTestCase
from urllib3 import HTTPConnectionPool, encode_multipart_formdata
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six import b, u
from urllib3.packages.six.moves.urllib.parse import urlencode
from urllib3.util import SKIP_HEADER, SKIPPABLE_HEADERS
from urllib3.util.retry import RequestHistory, Retry
from urllib3.util.timeout import Timeout
from .. import INVALID_SOURCE_ADDRESSES, TARPIT_HOST, VALID_SOURCE_ADDRESSES
from ..port_helpers import find_unused_port
@pytest.mark.parametrize('accept_encoding', ['Accept-Encoding', 'accept-encoding', b'Accept-Encoding', b'accept-encoding', None])
@pytest.mark.parametrize('host', ['Host', 'host', b'Host', b'host', None])
@pytest.mark.parametrize('user_agent', ['User-Agent', 'user-agent', b'User-Agent', b'user-agent', None])
@pytest.mark.parametrize('chunked', [True, False])
def test_skip_header(self, accept_encoding, host, user_agent, chunked):
    headers = {}
    if accept_encoding is not None:
        headers[accept_encoding] = SKIP_HEADER
    if host is not None:
        headers[host] = SKIP_HEADER
    if user_agent is not None:
        headers[user_agent] = SKIP_HEADER
    with HTTPConnectionPool(self.host, self.port) as pool:
        r = pool.request('GET', '/headers', headers=headers, chunked=chunked)
    request_headers = json.loads(r.data.decode('utf8'))
    if accept_encoding is None:
        assert 'Accept-Encoding' in request_headers
    else:
        assert accept_encoding not in request_headers
    if host is None:
        assert 'Host' in request_headers
    else:
        assert host not in request_headers
    if user_agent is None:
        assert 'User-Agent' in request_headers
    else:
        assert user_agent not in request_headers