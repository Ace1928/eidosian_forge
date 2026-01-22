import ssl
from . import http_server, ssl_certs, test_server
class TestingHTTPSServerMixin:

    def __init__(self, key_file, cert_file):
        self.key_file = key_file
        self.cert_file = cert_file

    def _get_ssl_request(self, sock, addr):
        """Wrap the socket with SSL"""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        if self.cert_file:
            ssl_context.load_cert_chain(self.cert_file, self.key_file)
        ssl_sock = ssl_context.wrap_socket(sock=sock, server_side=True, do_handshake_on_connect=False)
        return (ssl_sock, addr)

    def verify_request(self, request, client_address):
        """Verify the request.

        Return True if we should proceed with this request, False if we should
        not even touch a single byte in the socket !
        """
        serving = test_server.TestingTCPServerMixin.verify_request(self, request, client_address)
        if serving:
            try:
                request.do_handshake()
            except ssl.SSLError:
                return False
        return serving

    def ignored_exceptions_during_shutdown(self, e):
        base = test_server.TestingTCPServerMixin
        return base.ignored_exceptions_during_shutdown(self, e)