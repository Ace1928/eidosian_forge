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
def test_socket_object_attributes(self):
    """Ensures common socket attributes are exposed"""
    self.start_dummy_server()
    sock = socket.create_connection((self.host, self.port))
    with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
        assert ssock.fileno() is not None
        test_timeout = 10
        ssock.settimeout(test_timeout)
        assert ssock.gettimeout() == test_timeout
        assert ssock.socket.gettimeout() == test_timeout
        ssock.send(sample_request())
        response = consume_socket(ssock)
        validate_response(response)