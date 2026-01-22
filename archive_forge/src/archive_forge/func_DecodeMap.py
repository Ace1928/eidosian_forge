import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def DecodeMap(buffer, pos, end, message, field_dict):
    submsg = message_type._concrete_class()
    value = field_dict.get(key)
    if value is None:
        value = field_dict.setdefault(key, new_default(message))
    while 1:
        size, pos = local_DecodeVarint(buffer, pos)
        new_pos = pos + size
        if new_pos > end:
            raise _DecodeError('Truncated message.')
        submsg.Clear()
        if submsg._InternalParse(buffer, pos, new_pos) != new_pos:
            raise _DecodeError('Unexpected end-group tag.')
        if is_message_map:
            value[submsg.key].CopyFrom(submsg.value)
        else:
            value[submsg.key] = submsg.value
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos == end:
            return new_pos