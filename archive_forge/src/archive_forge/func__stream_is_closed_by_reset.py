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
def _stream_is_closed_by_reset(self, stream_id):
    """
        Returns ``True`` if the stream was closed by sending or receiving a
        RST_STREAM frame. Returns ``False`` otherwise.
        """
    return self._stream_closed_by(stream_id) in (StreamClosedBy.RECV_RST_STREAM, StreamClosedBy.SEND_RST_STREAM)