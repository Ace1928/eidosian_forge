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
def test_wrong_sni_hint(self):
    """
        Provides a wrong sni hint to validate an exception is thrown.
        """
    self.start_destination_server()
    self.start_proxy_server()
    sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
    with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
        with pytest.raises(Exception) as e:
            SSLTransport(proxy_sock, self.client_context, server_hostname='veryverywrong')
        assert e.type in [ssl.SSLError, ssl.CertificateError]