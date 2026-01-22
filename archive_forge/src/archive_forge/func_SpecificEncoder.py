import struct
from cloudsdk.google.protobuf.internal import wire_format
def SpecificEncoder(field_number, is_repeated, is_packed):
    local_struct_pack = struct.pack
    if is_packed:
        tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
        local_EncodeVarint = _EncodeVarint

        def EncodePackedField(write, value, deterministic):
            write(tag_bytes)
            local_EncodeVarint(write, len(value) * value_size, deterministic)
            for element in value:
                try:
                    write(local_struct_pack(format, element))
                except SystemError:
                    EncodeNonFiniteOrRaise(write, element)
        return EncodePackedField
    elif is_repeated:
        tag_bytes = TagBytes(field_number, wire_type)

        def EncodeRepeatedField(write, value, unused_deterministic=None):
            for element in value:
                write(tag_bytes)
                try:
                    write(local_struct_pack(format, element))
                except SystemError:
                    EncodeNonFiniteOrRaise(write, element)
        return EncodeRepeatedField
    else:
        tag_bytes = TagBytes(field_number, wire_type)

        def EncodeField(write, value, unused_deterministic=None):
            write(tag_bytes)
            try:
                write(local_struct_pack(format, value))
            except SystemError:
                EncodeNonFiniteOrRaise(write, value)
        return EncodeField