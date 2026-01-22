import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _ConvertToUnicode(memview):
    """Convert byte to unicode."""
    byte_str = memview.tobytes()
    try:
        value = str(byte_str, 'utf-8')
    except UnicodeDecodeError as e:
        e.reason = '%s in field: %s' % (e, key.full_name)
        raise
    return value