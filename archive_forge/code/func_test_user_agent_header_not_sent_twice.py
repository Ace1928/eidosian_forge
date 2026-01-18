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
@pytest.mark.parametrize('headers', [None, {}, {'User-Agent': 'key'}, {'user-agent': 'key'}, {b'uSeR-AgEnT': b'key'}, {b'user-agent': 'key'}])
@pytest.mark.parametrize('chunked', [True, False])
def test_user_agent_header_not_sent_twice(self, headers, chunked):
    with HTTPConnectionPool(self.host, self.port) as pool:
        r = pool.request('GET', '/headers', headers=headers, chunked=chunked)
        request_headers = json.loads(r.data.decode('utf8'))
        if not headers:
            assert request_headers['User-Agent'].startswith('python-urllib3/')
            assert 'key' not in request_headers['User-Agent']
        else:
            assert request_headers['User-Agent'] == 'key'