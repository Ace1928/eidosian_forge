import platform
import select
import socket
import ssl
import sys
import mock
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3.util import ssl_
from urllib3.util.ssltransport import SSLTransport
@pytest.mark.timeout(PER_TEST_TIMEOUT)
def test_start_closed_socket(self):
    """Errors generated from an unconnected socket should bubble up."""
    sock = socket.socket(socket.AF_INET)
    context = ssl.create_default_context()
    sock.close()
    with pytest.raises(OSError):
        SSLTransport(sock, context)