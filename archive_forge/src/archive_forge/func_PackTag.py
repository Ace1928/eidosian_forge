import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def PackTag(field_number, wire_type):
    """Returns an unsigned 32-bit integer that encodes the field number and
  wire type information in standard protocol message wire format.

  Args:
    field_number: Expected to be an integer in the range [1, 1 << 29)
    wire_type: One of the WIRETYPE_* constants.
  """
    if not 0 <= wire_type <= _WIRETYPE_MAX:
        raise message.EncodeError('Unknown wire type: %d' % wire_type)
    return field_number << TAG_TYPE_BITS | wire_type