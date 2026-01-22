from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _track_content_length(self, length, end_stream):
    """
        Update the expected content length in response to data being received.
        Validates that the appropriate amount of data is sent. Always updates
        the received data, but only validates the length against the
        content-length header if one was sent.

        :param length: The length of the body chunk received.
        :param end_stream: If this is the last body chunk received.
        """
    self._actual_content_length += length
    actual = self._actual_content_length
    expected = self._expected_content_length
    if expected is not None:
        if expected < actual:
            raise InvalidBodyLengthError(expected, actual)
        if end_stream and expected != actual:
            raise InvalidBodyLengthError(expected, actual)