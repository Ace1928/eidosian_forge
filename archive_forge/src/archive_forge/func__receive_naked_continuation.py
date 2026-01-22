import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def _receive_naked_continuation(self, frame):
    """
        A naked CONTINUATION frame has been received. This is always an error,
        but the type of error it is depends on the state of the stream and must
        transition the state of the stream, so we need to pass it to the
        appropriate stream.
        """
    stream = self._get_stream_by_id(frame.stream_id)
    stream.receive_continuation()
    assert False, 'Should not be reachable'