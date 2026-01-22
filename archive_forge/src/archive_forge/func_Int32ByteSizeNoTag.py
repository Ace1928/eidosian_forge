import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def Int32ByteSizeNoTag(int32):
    return _VarUInt64ByteSizeNoTag(18446744073709551615 & int32)