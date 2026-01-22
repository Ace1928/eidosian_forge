import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _RaiseInvalidWireType(buffer, pos, end):
    """Skip function for unknown wire types.  Raises an exception."""
    raise _DecodeError('Tag had invalid wire type.')