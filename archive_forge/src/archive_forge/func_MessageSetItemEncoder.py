import struct
from cloudsdk.google.protobuf.internal import wire_format
def MessageSetItemEncoder(field_number):
    """Encoder for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
    start_bytes = b''.join([TagBytes(1, wire_format.WIRETYPE_START_GROUP), TagBytes(2, wire_format.WIRETYPE_VARINT), _VarintBytes(field_number), TagBytes(3, wire_format.WIRETYPE_LENGTH_DELIMITED)])
    end_bytes = TagBytes(1, wire_format.WIRETYPE_END_GROUP)
    local_EncodeVarint = _EncodeVarint

    def EncodeField(write, value, deterministic):
        write(start_bytes)
        local_EncodeVarint(write, value.ByteSize(), deterministic)
        value._InternalSerialize(write, deterministic)
        return write(end_bytes)
    return EncodeField