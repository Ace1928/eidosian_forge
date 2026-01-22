import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class HeadersFrame(Padding, Priority, Frame):
    """
    The HEADERS frame carries name-value pairs. It is used to open a stream.
    HEADERS frames can be sent on a stream in the "open" or "half closed
    (remote)" states.

    The HeadersFrame class is actually basically a data frame in this
    implementation, because of the requirement to control the sizes of frames.
    A header block fragment that doesn't fit in an entire HEADERS frame needs
    to be followed with CONTINUATION frames. From the perspective of the frame
    building code the header block is an opaque data segment.
    """
    defined_flags = [Flag('END_STREAM', 1), Flag('END_HEADERS', 4), Flag('PADDED', 8), Flag('PRIORITY', 32)]
    type = 1
    stream_association = _STREAM_ASSOC_HAS_STREAM

    def __init__(self, stream_id, data=b'', **kwargs):
        super(HeadersFrame, self).__init__(stream_id, **kwargs)
        self.data = data

    def serialize_body(self):
        padding_data = self.serialize_padding_data()
        padding = b'\x00' * self.total_padding
        if 'PRIORITY' in self.flags:
            priority_data = self.serialize_priority_data()
        else:
            priority_data = b''
        return b''.join([padding_data, priority_data, self.data, padding])

    def parse_body(self, data):
        padding_data_length = self.parse_padding_data(data)
        data = data[padding_data_length:]
        if 'PRIORITY' in self.flags:
            priority_data_length = self.parse_priority_data(data)
        else:
            priority_data_length = 0
        self.body_len = len(data)
        self.data = data[priority_data_length:len(data) - self.total_padding].tobytes()
        if self.total_padding and self.total_padding >= self.body_len:
            raise InvalidPaddingError('Padding is too long.')