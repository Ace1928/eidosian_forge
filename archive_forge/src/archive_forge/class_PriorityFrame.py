import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class PriorityFrame(Priority, Frame):
    """
    The PRIORITY frame specifies the sender-advised priority of a stream. It
    can be sent at any time for an existing stream. This enables
    reprioritisation of existing streams.
    """
    defined_flags = []
    type = 2
    stream_association = _STREAM_ASSOC_HAS_STREAM

    def serialize_body(self):
        return self.serialize_priority_data()

    def parse_body(self, data):
        self.parse_priority_data(data)
        self.body_len = len(data)