import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_into_void(self):
    f = self._file
    start = self.first_range_start
    f.seek(start)
    f.seek(start + 40)
    f.seek(100)
    f.seek(125)