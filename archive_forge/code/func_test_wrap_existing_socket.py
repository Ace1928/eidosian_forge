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
def test_wrap_existing_socket(self):
    """Validates a single TLS layer can be established."""
    self.start_dummy_server()
    sock = socket.create_connection((self.host, self.port))
    with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
        assert ssock.version() is not None
        ssock.send(sample_request())
        response = consume_socket(ssock)
        validate_response(response)