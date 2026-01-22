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
@pytest.mark.skipif(sys.version_info < (3, 5), reason='requires python3.5 or higher')
class TlsInTlsTestCase(SocketDummyServerTestCase):
    """
    Creates a TLS in TLS tunnel by chaining a 'SocketProxyDummyServer' and a
    `SocketDummyServerTestCase`.

    Client will first connect to the proxy, who will then proxy any bytes send
    to the destination server. First TLS layer terminates at the proxy, second
    TLS layer terminates at the destination server.
    """

    @classmethod
    def setup_class(cls):
        cls.server_context, cls.client_context = server_client_ssl_contexts()

    @classmethod
    def start_proxy_server(cls):
        cls.proxy_server = SocketProxyDummyServer(cls.host, cls.port)
        cls.proxy_server.start_proxy_handler()

    @classmethod
    def teardown_class(cls):
        if hasattr(cls, 'proxy_server'):
            cls.proxy_server.teardown_class()
        super(TlsInTlsTestCase, cls).teardown_class()

    @classmethod
    def start_destination_server(cls):
        """
        Socket handler for the destination_server. Terminates the second TLS
        layer and send a basic HTTP response.
        """

        def socket_handler(listener):
            sock = listener.accept()[0]
            with cls.server_context.wrap_socket(sock, server_side=True) as ssock:
                request = consume_socket(ssock)
                validate_request(request)
                ssock.send(sample_response())
            sock.close()
        cls._start_server(socket_handler)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_tls_in_tls_tunnel(self):
        """
        Basic communication over the TLS in TLS tunnel.
        """
        self.start_destination_server()
        self.start_proxy_server()
        sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
        with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
            with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
                assert destination_sock.version() is not None
                destination_sock.send(sample_request())
                response = consume_socket(destination_sock)
                validate_response(response)

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