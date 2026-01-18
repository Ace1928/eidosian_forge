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
def test_unwrap_existing_socket(self):
    """
        Validates we can break up the TLS layer
        A full request/response is sent over TLS, and later over plain text.
        """

    def shutdown_handler(listener):
        sock = listener.accept()[0]
        ssl_sock = self.server_context.wrap_socket(sock, server_side=True)
        request = consume_socket(ssl_sock)
        validate_request(request)
        ssl_sock.sendall(sample_response())
        unwrapped_sock = ssl_sock.unwrap()
        request = consume_socket(unwrapped_sock)
        validate_request(request)
        unwrapped_sock.sendall(sample_response())
    self.start_dummy_server(shutdown_handler)
    sock = socket.create_connection((self.host, self.port))
    ssock = SSLTransport(sock, self.client_context, server_hostname='localhost')
    ssock.sendall(sample_request())
    response = consume_socket(ssock)
    validate_response(response)
    ssock.unwrap()
    sock.sendall(sample_request())
    response = consume_socket(sock)
    validate_response(response)