import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class TestCaseWithWebserver(tests.TestCaseWithTransport):
    """A support class that provides readonly urls that are http://.

    This is done by forcing the readonly server to be an http
    one. This will currently fail if the primary transport is not
    backed by regular disk files.
    """
    _protocol_version = None
    _url_protocol = 'http'

    def setUp(self):
        super().setUp()
        self.transport_readonly_server = http_server.HttpServer

    def create_transport_readonly_server(self):
        server = self.transport_readonly_server(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server