import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_read_all_ranges(self):
    f = self._file
    self.assertEqual(self.alpha, f.read())
    f.seek(100)
    self.assertEqual(self.alpha, f.read())
    self.assertEqual(126, f.tell())
    f.seek(126)
    self.assertEqual(b'A', f.read(1))
    f.seek(10, 1)
    self.assertEqual(b'LMN', f.read(3))