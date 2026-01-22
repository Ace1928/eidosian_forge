import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def UInt32ByteSize(field_number, uint32):
    return UInt64ByteSize(field_number, uint32)