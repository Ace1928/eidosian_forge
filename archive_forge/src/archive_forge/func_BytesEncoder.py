import struct
from cloudsdk.google.protobuf.internal import wire_format
def BytesEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a bytes field."""
    tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
    local_EncodeVarint = _EncodeVarint
    local_len = len
    assert not is_packed
    if is_repeated:

        def EncodeRepeatedField(write, value, deterministic):
            for element in value:
                write(tag)
                local_EncodeVarint(write, local_len(element), deterministic)
                write(element)
        return EncodeRepeatedField
    else:

        def EncodeField(write, value, deterministic):
            write(tag)
            local_EncodeVarint(write, local_len(value), deterministic)
            return write(value)
        return EncodeField