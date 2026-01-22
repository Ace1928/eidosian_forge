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
def _receive_data_frame(self, frame):
    """
        Receive a data frame on the connection.
        """
    flow_controlled_length = frame.flow_controlled_length
    events = self.state_machine.process_input(ConnectionInputs.RECV_DATA)
    self._inbound_flow_control_window_manager.window_consumed(flow_controlled_length)
    try:
        stream = self._get_stream_by_id(frame.stream_id)
        frames, stream_events = stream.receive_data(frame.data, 'END_STREAM' in frame.flags, flow_controlled_length)
    except StreamClosedError as e:
        return self._handle_data_on_closed_stream(events, e, frame)
    return (frames, events + stream_events)