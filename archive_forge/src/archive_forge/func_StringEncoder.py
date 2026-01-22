import struct
from cloudsdk.google.protobuf.internal import wire_format
def StringEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a string field."""
    tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
    local_EncodeVarint = _EncodeVarint
    local_len = len
    assert not is_packed
    if is_repeated:

        def EncodeRepeatedField(write, value, deterministic):
            for element in value:
                encoded = element.encode('utf-8')
                write(tag)
                local_EncodeVarint(write, local_len(encoded), deterministic)
                write(encoded)
        return EncodeRepeatedField
    else:

        def EncodeField(write, value, deterministic):
            encoded = value.encode('utf-8')
            write(tag)
            local_EncodeVarint(write, local_len(encoded), deterministic)
            return write(encoded)
        return EncodeField