import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileMultipleRangesQuotedBoundaries(TestRangeFileMultipleRanges):
    """Perform the same tests as TestRangeFileMultipleRanges, but uses
    an angle-bracket quoted boundary string like IIS 6.0 and 7.0
    (but not IIS 5, which breaks the RFC in a different way
    by using square brackets, not angle brackets)

    This reveals a bug caused by

    - The bad implementation of RFC 822 unquoting in Python (angles are not
      quotes), coupled with

    - The bad implementation of RFC 2046 in IIS (angles are not permitted chars
      in boundary lines).

    """
    _boundary_trimmed = b'q1w2e3r4t5y6u7i8o9p0zaxscdvfbgnhmjklkl'
    boundary = b'<' + _boundary_trimmed + b'>'

    def set_file_boundary(self):
        self._file.set_boundary(self._boundary_trimmed)