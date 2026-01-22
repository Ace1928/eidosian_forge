import struct
from cloudsdk.google.protobuf.internal import wire_format
def GroupEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a group field."""
    start_tag = TagBytes(field_number, wire_format.WIRETYPE_START_GROUP)
    end_tag = TagBytes(field_number, wire_format.WIRETYPE_END_GROUP)
    assert not is_packed
    if is_repeated:

        def EncodeRepeatedField(write, value, deterministic):
            for element in value:
                write(start_tag)
                element._InternalSerialize(write, deterministic)
                write(end_tag)
        return EncodeRepeatedField
    else:

        def EncodeField(write, value, deterministic):
            write(start_tag)
            value._InternalSerialize(write, deterministic)
            return write(end_tag)
        return EncodeField