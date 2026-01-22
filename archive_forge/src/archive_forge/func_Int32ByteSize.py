import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def Int32ByteSize(field_number, int32):
    return Int64ByteSize(field_number, int32)