import struct
from cloudsdk.google.protobuf.internal import wire_format
def GroupSizer(field_number, is_repeated, is_packed):
    """Returns a sizer for a group field."""
    tag_size = _TagSize(field_number) * 2
    assert not is_packed
    if is_repeated:

        def RepeatedFieldSize(value):
            result = tag_size * len(value)
            for element in value:
                result += element.ByteSize()
            return result
        return RepeatedFieldSize
    else:

        def FieldSize(value):
            return tag_size + value.ByteSize()
        return FieldSize