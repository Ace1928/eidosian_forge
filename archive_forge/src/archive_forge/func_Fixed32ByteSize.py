import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def Fixed32ByteSize(field_number, fixed32):
    return TagByteSize(field_number) + 4