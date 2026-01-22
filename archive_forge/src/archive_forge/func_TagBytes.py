import struct
from cloudsdk.google.protobuf.internal import wire_format
def TagBytes(field_number, wire_type):
    """Encode the given tag and return the bytes.  Only called at startup."""
    return bytes(_VarintBytes(wire_format.PackTag(field_number, wire_type)))