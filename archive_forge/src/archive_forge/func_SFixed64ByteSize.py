import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def SFixed64ByteSize(field_number, sfixed64):
    return TagByteSize(field_number) + 8