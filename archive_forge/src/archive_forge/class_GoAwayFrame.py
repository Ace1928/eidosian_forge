import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class GoAwayFrame(Frame):
    """
    The GOAWAY frame informs the remote peer to stop creating streams on this
    connection. It can be sent from the client or the server. Once sent, the
    sender will ignore frames sent on new streams for the remainder of the
    connection.
    """
    defined_flags = []
    type = 7
    stream_association = _STREAM_ASSOC_NO_STREAM

    def __init__(self, stream_id=0, last_stream_id=0, error_code=0, additional_data=b'', **kwargs):
        super(GoAwayFrame, self).__init__(stream_id, **kwargs)
        self.last_stream_id = last_stream_id
        self.error_code = error_code
        self.additional_data = additional_data

    def serialize_body(self):
        data = _STRUCT_LL.pack(self.last_stream_id & 2147483647, self.error_code)
        data += self.additional_data
        return data

    def parse_body(self, data):
        try:
            self.last_stream_id, self.error_code = _STRUCT_LL.unpack(data[:8])
        except struct.error:
            raise InvalidFrameError('Invalid GOAWAY body.')
        self.body_len = len(data)
        if len(data) > 8:
            self.additional_data = data[8:].tobytes()