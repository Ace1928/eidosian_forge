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
@pytest.mark.skipif(platform.system() == 'Windows', reason='Skipping windows due to text makefile support')
@pytest.mark.timeout(PER_TEST_TIMEOUT)
def test_tls_in_tls_makefile_rw_text(self):
    """
        Creates a separate buffer for reading and writing using text mode and
        utf-8 encoding.
        """
    self.start_destination_server()
    self.start_proxy_server()
    sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
    with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
        with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
            read = destination_sock.makefile('r', encoding='utf-8')
            write = destination_sock.makefile('w', encoding='utf-8')
            write.write(sample_request(binary=False))
            write.flush()
            response = read.read()
            if '\r' not in response:
                response = response.replace('\n', '\r\n')
            validate_response(response, binary=False)