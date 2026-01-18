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
def test_tls_in_tls_recv_into_unbuffered(self):
    """
        Valides recv_into without a preallocated buffer.
        """
    self.start_destination_server()
    self.start_proxy_server()
    sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
    with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
        with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
            destination_sock.send(sample_request())
            response = destination_sock.recv_into(None)
            validate_response(response)