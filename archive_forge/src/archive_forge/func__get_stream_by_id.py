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
def _get_stream_by_id(self, stream_id):
    """
        Gets a stream by its stream ID. Raises NoSuchStreamError if the stream
        ID does not correspond to a known stream and is higher than the current
        maximum: raises if it is lower than the current maximum.

        .. versionchanged:: 2.0.0
           Removed this function from the public API.
        """
    try:
        return self.streams[stream_id]
    except KeyError:
        outbound = self._stream_id_is_outbound(stream_id)
        highest_stream_id = self.highest_outbound_stream_id if outbound else self.highest_inbound_stream_id
        if stream_id > highest_stream_id:
            raise NoSuchStreamError(stream_id)
        else:
            raise StreamClosedError(stream_id)