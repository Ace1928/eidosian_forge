import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class ReadSocket:
    """A socket-like object that can be given a predefined content."""

    def __init__(self, data):
        self.readfile = BytesIO(data)

    def makefile(self, mode='r', bufsize=None):
        return self.readfile