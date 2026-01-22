import struct
from cloudsdk.google.protobuf.internal import wire_format
def _ModifiedEncoder(wire_type, encode_value, compute_value_size, modify_value):
    """Like SimpleEncoder but additionally invokes modify_value on every value
  before passing it to encode_value.  Usually modify_value is ZigZagEncode."""

    def SpecificEncoder(field_number, is_repeated, is_packed):
        if is_packed:
            tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
            local_EncodeVarint = _EncodeVarint

            def EncodePackedField(write, value, deterministic):
                write(tag_bytes)
                size = 0
                for element in value:
                    size += compute_value_size(modify_value(element))
                local_EncodeVarint(write, size, deterministic)
                for element in value:
                    encode_value(write, modify_value(element), deterministic)
            return EncodePackedField
        elif is_repeated:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeRepeatedField(write, value, deterministic):
                for element in value:
                    write(tag_bytes)
                    encode_value(write, modify_value(element), deterministic)
            return EncodeRepeatedField
        else:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeField(write, value, deterministic):
                write(tag_bytes)
                return encode_value(write, modify_value(value), deterministic)
            return EncodeField
    return SpecificEncoder