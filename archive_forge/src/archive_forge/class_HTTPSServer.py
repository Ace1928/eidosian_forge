import ssl
from . import http_server, ssl_certs, test_server
class HTTPSServer(http_server.HttpServer):
    _url_protocol = 'https'
    http_server_class = {'HTTP/1.0': TestingHTTPSServer, 'HTTP/1.1': TestingThreadingHTTPSServer}

    def __init__(self, request_handler=http_server.TestingHTTPRequestHandler, protocol_version=None, key_file=ssl_certs.build_path('server_without_pass.key'), cert_file=ssl_certs.build_path('server.crt')):
        http_server.HttpServer.__init__(self, request_handler=request_handler, protocol_version=protocol_version)
        self.key_file = key_file
        self.cert_file = cert_file
        self.temp_files = []

    def create_server(self):
        return self.server_class((self.host, self.port), self.request_handler_class, self, self.key_file, self.cert_file)