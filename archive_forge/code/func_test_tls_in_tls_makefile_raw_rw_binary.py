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
@pytest.mark.parametrize('buffering', [None, 0])
def test_tls_in_tls_makefile_raw_rw_binary(self, buffering):
    """
        Uses makefile with read, write and binary modes without buffering.
        """
    self.start_destination_server()
    self.start_proxy_server()
    sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
    with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
        with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
            file = destination_sock.makefile('rwb', buffering)
            file.write(sample_request())
            file.flush()
            response = bytearray(65536)
            wrote = file.readinto(response)
            assert wrote is not None
            str_response = response.decode('utf-8').rstrip('\x00')
            validate_response(str_response, binary=False)
            file.close()