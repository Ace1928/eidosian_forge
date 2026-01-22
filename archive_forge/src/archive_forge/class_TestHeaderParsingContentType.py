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
class TestHeaderParsingContentType(SocketDummyServerTestCase):

    def _test_okay_header_parsing(self, header):
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n' + header + b'\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with LogRecorder() as logs:
                pool.request('GET', '/')
            for record in logs:
                assert 'Failed to parse headers' not in record.msg

    def test_header_text_plain(self):
        self._test_okay_header_parsing(b'Content-type: text/plain')

    def test_header_message_rfc822(self):
        self._test_okay_header_parsing(b'Content-type: message/rfc822')