import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def UnpackTag(tag):
    """The inverse of PackTag().  Given an unsigned 32-bit number,
  returns a (field_number, wire_type) tuple.
  """
    return (tag >> TAG_TYPE_BITS, tag & TAG_TYPE_MASK)