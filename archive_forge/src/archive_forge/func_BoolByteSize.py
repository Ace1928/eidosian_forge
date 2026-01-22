import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def BoolByteSize(field_number, b):
    return TagByteSize(field_number) + 1