import struct
from cloudsdk.google.protobuf.internal import wire_format
def _SimpleSizer(compute_value_size):
    """A sizer which uses the function compute_value_size to compute the size of
  each value.  Typically compute_value_size is _VarintSize."""

    def SpecificSizer(field_number, is_repeated, is_packed):
        tag_size = _TagSize(field_number)
        if is_packed:
            local_VarintSize = _VarintSize

            def PackedFieldSize(value):
                result = 0
                for element in value:
                    result += compute_value_size(element)
                return result + local_VarintSize(result) + tag_size
            return PackedFieldSize
        elif is_repeated:

            def RepeatedFieldSize(value):
                result = tag_size * len(value)
                for element in value:
                    result += compute_value_size(element)
                return result
            return RepeatedFieldSize
        else:

            def FieldSize(value):
                return tag_size + compute_value_size(value)
            return FieldSize
    return SpecificSizer