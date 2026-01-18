import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_at_range_end(self):
    """Test seek behavior at range end."""
    f = self._file
    f.seek(25 + 25)
    f.seek(100 + 25)
    f.seek(126 + 25)