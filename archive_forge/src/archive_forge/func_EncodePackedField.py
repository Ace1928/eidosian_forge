import struct
from cloudsdk.google.protobuf.internal import wire_format
def EncodePackedField(write, value, deterministic):
    write(tag_bytes)
    local_EncodeVarint(write, len(value) * value_size, deterministic)
    for element in value:
        try:
            write(local_struct_pack(format, element))
        except SystemError:
            EncodeNonFiniteOrRaise(write, element)