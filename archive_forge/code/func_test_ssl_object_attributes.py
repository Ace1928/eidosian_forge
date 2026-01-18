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
def test_ssl_object_attributes(self):
    """Ensures common ssl attributes are exposed"""
    self.start_dummy_server()
    sock = socket.create_connection((self.host, self.port))
    with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
        cipher = ssock.cipher()
        assert type(cipher) == tuple
        assert ssock.selected_alpn_protocol() is None
        assert ssock.selected_npn_protocol() is None
        shared_ciphers = ssock.shared_ciphers()
        assert type(shared_ciphers) == list
        assert len(shared_ciphers) > 0
        assert ssock.compression() is None
        validate_peercert(ssock)
        ssock.send(sample_request())
        response = consume_socket(ssock)
        validate_response(response)