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
def _receive_headers_frame(self, frame):
    """
        Receive a headers frame on the connection.
        """
    if frame.stream_id not in self.streams:
        max_open_streams = self.local_settings.max_concurrent_streams
        if self.open_inbound_streams + 1 > max_open_streams:
            raise TooManyStreamsError('Max outbound streams is %d, %d open' % (max_open_streams, self.open_outbound_streams))
    headers = _decode_headers(self.decoder, frame.data)
    events = self.state_machine.process_input(ConnectionInputs.RECV_HEADERS)
    stream = self._get_or_create_stream(frame.stream_id, AllowedStreamIDs(not self.config.client_side))
    frames, stream_events = stream.receive_headers(headers, 'END_STREAM' in frame.flags, self.config.header_encoding)
    if 'PRIORITY' in frame.flags:
        p_frames, p_events = self._receive_priority_frame(frame)
        stream_events[0].priority_updated = p_events[0]
        stream_events.extend(p_events)
        assert not p_frames
    return (frames, events + stream_events)