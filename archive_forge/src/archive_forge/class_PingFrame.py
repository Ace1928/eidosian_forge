import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class PingFrame(Frame):
    """
    The PING frame is a mechanism for measuring a minimal round-trip time from
    the sender, as well as determining whether an idle connection is still
    functional. PING frames can be sent from any endpoint.
    """
    defined_flags = [Flag('ACK', 1)]
    type = 6
    stream_association = _STREAM_ASSOC_NO_STREAM

    def __init__(self, stream_id=0, opaque_data=b'', **kwargs):
        super(PingFrame, self).__init__(stream_id, **kwargs)
        self.opaque_data = opaque_data

    def serialize_body(self):
        if len(self.opaque_data) > 8:
            raise InvalidFrameError('PING frame may not have more than 8 bytes of data, got %s' % self.opaque_data)
        data = self.opaque_data
        data += b'\x00' * (8 - len(self.opaque_data))
        return data

    def parse_body(self, data):
        if len(data) != 8:
            raise InvalidFrameError('PING frame must have 8 byte length: got %s' % len(data))
        self.opaque_data = data.tobytes()
        self.body_len = 8