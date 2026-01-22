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
def _receive_rst_stream_frame(self, frame):
    """
        Receive a RST_STREAM frame on the connection.
        """
    events = self.state_machine.process_input(ConnectionInputs.RECV_RST_STREAM)
    try:
        stream = self._get_stream_by_id(frame.stream_id)
    except NoSuchStreamError:
        stream_frames = []
        stream_events = []
    else:
        stream_frames, stream_events = stream.stream_reset(frame)
    return (stream_frames, events + stream_events)