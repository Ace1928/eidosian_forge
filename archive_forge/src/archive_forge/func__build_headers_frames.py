from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _build_headers_frames(self, headers, encoder, first_frame, hdr_validation_flags):
    """
        Helper method to build headers or push promise frames.
        """
    if self.config.normalize_outbound_headers:
        headers = normalize_outbound_headers(headers, hdr_validation_flags)
    if self.config.validate_outbound_headers:
        headers = validate_outbound_headers(headers, hdr_validation_flags)
    encoded_headers = encoder.encode(headers)
    header_blocks = [encoded_headers[i:i + self.max_outbound_frame_size] for i in range(0, len(encoded_headers), self.max_outbound_frame_size)]
    frames = []
    first_frame.data = header_blocks[0]
    frames.append(first_frame)
    for block in header_blocks[1:]:
        cf = ContinuationFrame(self.stream_id)
        cf.data = block
        frames.append(cf)
    frames[-1].flags.add('END_HEADERS')
    return frames