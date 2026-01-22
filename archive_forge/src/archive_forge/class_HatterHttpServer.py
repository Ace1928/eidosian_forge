import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
class HatterHttpServer(http_server.HttpServer):
    """A server giving all sort of crazy responses (like Alice's Hatter).

    This is used to test various error cases in the webdav client.
    """

    def __init__(self):
        super().__init__(CannedRequestHandler, protocol_version='HTTP/1.1')
        self.canned_response = None