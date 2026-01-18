import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_read_past_end_of_range(self):
    f = self._file
    if f._size == -1:
        raise tests.TestNotApplicable("Can't check an unknown size")
    start = self.first_range_start
    f.seek(start + 20)
    self.assertRaises(errors.InvalidRange, f.read, 10)