import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _SkipFixed32(buffer, pos, end):
    """Skip a fixed32 value.  Returns the new position."""
    pos += 4
    if pos > end:
        raise _DecodeError('Truncated message.')
    return pos