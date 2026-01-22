import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _DecodeFixed64(buffer, pos):
    """Decode a fixed64."""
    new_pos = pos + 8
    return (struct.unpack('<Q', buffer[pos:new_pos])[0], new_pos)