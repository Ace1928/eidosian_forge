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
def test_ssl_custom_validation_failure_terminates(self, tmpdir):
    """
        Ensure that the underlying socket is terminated if custom validation fails.
        """
    server_closed = Event()

    def is_closed_socket(sock):
        try:
            sock.settimeout(SHORT_TIMEOUT)
            sock.recv(1)
        except (OSError, socket.error):
            return True
        return False

    def socket_handler(listener):
        sock = listener.accept()[0]
        try:
            _ = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
        except ssl.SSLError as e:
            assert 'alert unknown ca' in str(e)
            if is_closed_socket(sock):
                server_closed.set()
    self._start_server(socket_handler)
    other_ca = trustme.CA()
    other_ca_path = str(tmpdir / 'ca.pem')
    other_ca.cert_pem.write_to_path(other_ca_path)
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='REQUIRED', ca_certs=other_ca_path) as pool:
        with pytest.raises(SSLError):
            pool.request('GET', '/', retries=False, timeout=LONG_TIMEOUT)
    assert server_closed.wait(LONG_TIMEOUT), 'The socket was not terminated'