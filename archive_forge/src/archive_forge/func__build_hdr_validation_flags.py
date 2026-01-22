from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _build_hdr_validation_flags(self, events):
    """
        Constructs a set of header validation flags for use when normalizing
        and validating header blocks.
        """
    is_trailer = isinstance(events[0], (_TrailersSent, TrailersReceived))
    is_response_header = isinstance(events[0], (_ResponseSent, ResponseReceived, InformationalResponseReceived))
    is_push_promise = isinstance(events[0], (PushedStreamReceived, _PushedRequestSent))
    return HeaderValidationFlags(is_client=self.state_machine.client, is_trailer=is_trailer, is_response_header=is_response_header, is_push_promise=is_push_promise)