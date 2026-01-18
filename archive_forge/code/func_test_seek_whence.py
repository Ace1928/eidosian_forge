import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_whence(self):
    """Test the seek whence parameter values."""
    f = response.RangeFile('foo', BytesIO(b'abc'))
    f.set_range(0, 3)
    f.seek(0)
    f.seek(1, 1)
    f.seek(-1, 2)
    self.assertRaises(ValueError, f.seek, 0, 14)