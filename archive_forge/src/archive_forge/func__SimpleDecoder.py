import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _SimpleDecoder(wire_type, decode_value):
    """Return a constructor for a decoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type.
      decode_value:  A function which decodes an individual value, e.g.
        _DecodeVarint()
  """

    def SpecificDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
        if is_packed:
            local_DecodeVarint = _DecodeVarint

            def DecodePackedField(buffer, pos, end, message, field_dict):
                value = field_dict.get(key)
                if value is None:
                    value = field_dict.setdefault(key, new_default(message))
                endpoint, pos = local_DecodeVarint(buffer, pos)
                endpoint += pos
                if endpoint > end:
                    raise _DecodeError('Truncated message.')
                while pos < endpoint:
                    element, pos = decode_value(buffer, pos)
                    value.append(element)
                if pos > endpoint:
                    del value[-1]
                    raise _DecodeError('Packed element was truncated.')
                return pos
            return DecodePackedField
        elif is_repeated:
            tag_bytes = encoder.TagBytes(field_number, wire_type)
            tag_len = len(tag_bytes)

            def DecodeRepeatedField(buffer, pos, end, message, field_dict):
                value = field_dict.get(key)
                if value is None:
                    value = field_dict.setdefault(key, new_default(message))
                while 1:
                    element, new_pos = decode_value(buffer, pos)
                    value.append(element)
                    pos = new_pos + tag_len
                    if buffer[new_pos:pos] != tag_bytes or new_pos >= end:
                        if new_pos > end:
                            raise _DecodeError('Truncated message.')
                        return new_pos
            return DecodeRepeatedField
        else:

            def DecodeField(buffer, pos, end, message, field_dict):
                new_value, pos = decode_value(buffer, pos)
                if pos > end:
                    raise _DecodeError('Truncated message.')
                if clear_if_default and (not new_value):
                    field_dict.pop(key, None)
                else:
                    field_dict[key] = new_value
                return pos
            return DecodeField
    return SpecificDecoder