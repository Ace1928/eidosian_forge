from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _initialize_content_length(self, headers):
    """
        Checks the headers for a content-length header and initializes the
        _expected_content_length field from it. It's not an error for no
        Content-Length header to be present.
        """
    if self.request_method == b'HEAD':
        self._expected_content_length = 0
        return
    for n, v in headers:
        if n == b'content-length':
            try:
                self._expected_content_length = int(v, 10)
            except ValueError:
                raise ProtocolError('Invalid content-length header: %s' % v)
            return