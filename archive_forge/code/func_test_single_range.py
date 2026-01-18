import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_single_range(self):
    out = self.get_response(_single_range_response)
    out.seek(100)
    self.assertEqual(_single_range_response[2], out.read(100))