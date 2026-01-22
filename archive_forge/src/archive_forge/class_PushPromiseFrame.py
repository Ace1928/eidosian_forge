import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class PushPromiseFrame(Padding, Frame):
    """
    The PUSH_PROMISE frame is used to notify the peer endpoint in advance of
    streams the sender intends to initiate.
    """
    defined_flags = [Flag('END_HEADERS', 4), Flag('PADDED', 8)]
    type = 5
    stream_association = _STREAM_ASSOC_HAS_STREAM

    def __init__(self, stream_id, promised_stream_id=0, data=b'', **kwargs):
        super(PushPromiseFrame, self).__init__(stream_id, **kwargs)
        self.promised_stream_id = promised_stream_id
        self.data = data

    def serialize_body(self):
        padding_data = self.serialize_padding_data()
        padding = b'\x00' * self.total_padding
        data = _STRUCT_L.pack(self.promised_stream_id)
        return b''.join([padding_data, data, self.data, padding])

    def parse_body(self, data):
        padding_data_length = self.parse_padding_data(data)
        try:
            self.promised_stream_id = _STRUCT_L.unpack(data[padding_data_length:padding_data_length + 4])[0]
        except struct.error:
            raise InvalidFrameError('Invalid PUSH_PROMISE body')
        self.data = data[padding_data_length + 4:].tobytes()
        self.body_len = len(data)
        if self.total_padding and self.total_padding >= self.body_len:
            raise InvalidPaddingError('Padding is too long.')