import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class FakeHTTPConnection(urllib.HTTPConnection):

    def __init__(self, sock):
        urllib.HTTPConnection.__init__(self, 'localhost')
        self.sock = sock

    def send(self, str):
        """Ignores the writes on the socket."""
        pass