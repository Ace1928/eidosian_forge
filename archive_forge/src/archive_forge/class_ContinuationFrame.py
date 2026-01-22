import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class ContinuationFrame(Frame):
    """
    The CONTINUATION frame is used to continue a sequence of header block
    fragments. Any number of CONTINUATION frames can be sent on an existing
    stream, as long as the preceding frame on the same stream is one of
    HEADERS, PUSH_PROMISE or CONTINUATION without the END_HEADERS flag set.

    Much like the HEADERS frame, hyper treats this as an opaque data frame with
    different flags and a different type.
    """
    defined_flags = [Flag('END_HEADERS', 4)]
    type = 9
    stream_association = _STREAM_ASSOC_HAS_STREAM

    def __init__(self, stream_id, data=b'', **kwargs):
        super(ContinuationFrame, self).__init__(stream_id, **kwargs)
        self.data = data

    def serialize_body(self):
        return self.data

    def parse_body(self, data):
        self.data = data.tobytes()
        self.body_len = len(data)