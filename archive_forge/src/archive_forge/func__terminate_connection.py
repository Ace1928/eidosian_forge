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
def _terminate_connection(self, error_code):
    """
        Terminate the connection early. Used in error handling blocks to send
        GOAWAY frames.
        """
    f = GoAwayFrame(0)
    f.last_stream_id = self.highest_inbound_stream_id
    f.error_code = error_code
    self.state_machine.process_input(ConnectionInputs.SEND_GOAWAY)
    self._prepare_for_sending([f])