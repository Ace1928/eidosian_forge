import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def _boundary_line(self):
    """Helper to build the formatted boundary line."""
    return b'--' + self.boundary + b'\r\n'