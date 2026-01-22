import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _DecodeUnknownField(buffer, pos, wire_type):
    """Decode a unknown field.  Returns the UnknownField and new position."""
    if wire_type == wire_format.WIRETYPE_VARINT:
        data, pos = _DecodeVarint(buffer, pos)
    elif wire_type == wire_format.WIRETYPE_FIXED64:
        data, pos = _DecodeFixed64(buffer, pos)
    elif wire_type == wire_format.WIRETYPE_FIXED32:
        data, pos = _DecodeFixed32(buffer, pos)
    elif wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
        size, pos = _DecodeVarint(buffer, pos)
        data = buffer[pos:pos + size].tobytes()
        pos += size
    elif wire_type == wire_format.WIRETYPE_START_GROUP:
        data, pos = _DecodeUnknownFieldSet(buffer, pos)
    elif wire_type == wire_format.WIRETYPE_END_GROUP:
        return (0, -1)
    else:
        raise _DecodeError('Wrong wire type in tag.')
    return (data, pos)