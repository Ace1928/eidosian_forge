import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _SkipGroup(buffer, pos, end):
    """Skip sub-group.  Returns the new position."""
    while 1:
        tag_bytes, pos = ReadTag(buffer, pos)
        new_pos = SkipField(buffer, pos, end, tag_bytes)
        if new_pos == -1:
            return pos
        pos = new_pos