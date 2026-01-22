import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileSizeKnown(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for a whole file whose size is known."""

    def setUp(self):
        super().setUp()
        self._file = response.RangeFile('Whole_file_size_known', BytesIO(self.alpha))
        self._file.set_range(0, len(self.alpha))
        self.first_range_start = 0