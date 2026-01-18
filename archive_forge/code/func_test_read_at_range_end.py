import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_read_at_range_end(self):
    f = self._file
    self.assertEqual(self.alpha, f.read())
    self.assertEqual(self.alpha, f.read())
    self.assertEqual(self.alpha.upper(), f.read())
    self.assertRaises(errors.InvalidHttpResponse, f.read, 1)