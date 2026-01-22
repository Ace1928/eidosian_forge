import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def MessageDecoder(field_number, is_repeated, is_packed, key, new_default):
    """Returns a decoder for a message field."""
    local_DecodeVarint = _DecodeVarint
    assert not is_packed
    if is_repeated:
        tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
        tag_len = len(tag_bytes)

        def DecodeRepeatedField(buffer, pos, end, message, field_dict):
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            while 1:
                size, pos = local_DecodeVarint(buffer, pos)
                new_pos = pos + size
                if new_pos > end:
                    raise _DecodeError('Truncated message.')
                if value.add()._InternalParse(buffer, pos, new_pos) != new_pos:
                    raise _DecodeError('Unexpected end-group tag.')
                pos = new_pos + tag_len
                if buffer[new_pos:pos] != tag_bytes or new_pos == end:
                    return new_pos
        return DecodeRepeatedField
    else:

        def DecodeField(buffer, pos, end, message, field_dict):
            value = field_dict.get(key)
            if value is None:
                value = field_dict.setdefault(key, new_default(message))
            size, pos = local_DecodeVarint(buffer, pos)
            new_pos = pos + size
            if new_pos > end:
                raise _DecodeError('Truncated message.')
            if value._InternalParse(buffer, pos, new_pos) != new_pos:
                raise _DecodeError('Unexpected end-group tag.')
            return new_pos
        return DecodeField