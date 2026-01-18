import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_outside_single_range(self):
    f = self._file
    if f._size == -1 or f._boundary is not None:
        raise tests.TestNotApplicable('Needs a fully defined range')
    self.assertRaises(errors.InvalidRange, f.seek, self.first_range_start + 27)