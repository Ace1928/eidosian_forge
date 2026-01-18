import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_read_zero(self):
    f = self._file
    self.assertEqual(b'', f.read(0))
    f.seek(10, 1)
    self.assertEqual(b'', f.read(0))