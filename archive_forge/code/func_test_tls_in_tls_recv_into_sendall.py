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
def test_tls_in_tls_recv_into_sendall(self):
    """
        Valides recv_into and sendall also work as expected. Other tests are
        using recv/send.
        """
    self.start_destination_server()
    self.start_proxy_server()
    sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
    with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
        with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
            destination_sock.sendall(sample_request())
            response = bytearray(65536)
            destination_sock.recv_into(response)
            str_response = response.decode('utf-8').rstrip('\x00')
            validate_response(str_response, binary=False)