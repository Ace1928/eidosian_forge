import struct
from cloudsdk.google.protobuf.internal import wire_format
def EncodeField(write, value, unused_deterministic=None):
    write(tag_bytes)
    try:
        write(local_struct_pack(format, value))
    except SystemError:
        EncodeNonFiniteOrRaise(write, value)