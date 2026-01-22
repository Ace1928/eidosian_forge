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
def _receive_goaway_frame(self, frame):
    """
        Receive a GOAWAY frame on the connection.
        """
    events = self.state_machine.process_input(ConnectionInputs.RECV_GOAWAY)
    self.clear_outbound_data_buffer()
    new_event = ConnectionTerminated()
    new_event.error_code = _error_code_from_int(frame.error_code)
    new_event.last_stream_id = frame.last_stream_id
    new_event.additional_data = frame.additional_data if frame.additional_data else None
    events.append(new_event)
    return ([], events)