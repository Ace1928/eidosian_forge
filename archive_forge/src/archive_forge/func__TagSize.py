import struct
from cloudsdk.google.protobuf.internal import wire_format
def _TagSize(field_number):
    """Returns the number of bytes required to serialize a tag with this field
  number."""
    return _VarintSize(wire_format.PackTag(field_number, 0))