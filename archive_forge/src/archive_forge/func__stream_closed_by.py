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
def _stream_closed_by(self, stream_id):
    """
        Returns how the stream was closed.

        The return value will be either a member of
        ``h2.stream.StreamClosedBy`` or ``None``. If ``None``, the stream was
        closed implicitly by the peer opening a stream with a higher stream ID
        before opening this one.
        """
    if stream_id in self.streams:
        return self.streams[stream_id].closed_by
    if stream_id in self._closed_streams:
        return self._closed_streams[stream_id]
    return None