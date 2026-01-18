import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_unbounded_read_after_seek(self):
    f = self._file
    f.seek(24, 1)
    self.assertEqual(b'yz', f.read())