import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _SkipVarint(buffer, pos, end):
    """Skip a varint value.  Returns the new position."""
    while ord(buffer[pos:pos + 1].tobytes()) & 128:
        pos += 1
    pos += 1
    if pos > end:
        raise _DecodeError('Truncated message.')
    return pos