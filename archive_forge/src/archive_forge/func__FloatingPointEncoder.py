import struct
from cloudsdk.google.protobuf.internal import wire_format
def _FloatingPointEncoder(wire_type, format):
    """Return a constructor for an encoder for float fields.

  This is like StructPackEncoder, but catches errors that may be due to
  passing non-finite floating-point values to struct.pack, and makes a
  second attempt to encode those values.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """
    value_size = struct.calcsize(format)
    if value_size == 4:

        def EncodeNonFiniteOrRaise(write, value):
            if value == _POS_INF:
                write(b'\x00\x00\x80\x7f')
            elif value == _NEG_INF:
                write(b'\x00\x00\x80\xff')
            elif value != value:
                write(b'\x00\x00\xc0\x7f')
            else:
                raise
    elif value_size == 8:

        def EncodeNonFiniteOrRaise(write, value):
            if value == _POS_INF:
                write(b'\x00\x00\x00\x00\x00\x00\xf0\x7f')
            elif value == _NEG_INF:
                write(b'\x00\x00\x00\x00\x00\x00\xf0\xff')
            elif value != value:
                write(b'\x00\x00\x00\x00\x00\x00\xf8\x7f')
            else:
                raise
    else:
        raise ValueError("Can't encode floating-point values that are %d bytes long (only 4 or 8)" % value_size)

    def SpecificEncoder(field_number, is_repeated, is_packed):
        local_struct_pack = struct.pack
        if is_packed:
            tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
            local_EncodeVarint = _EncodeVarint

            def EncodePackedField(write, value, deterministic):
                write(tag_bytes)
                local_EncodeVarint(write, len(value) * value_size, deterministic)
                for element in value:
                    try:
                        write(local_struct_pack(format, element))
                    except SystemError:
                        EncodeNonFiniteOrRaise(write, element)
            return EncodePackedField
        elif is_repeated:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeRepeatedField(write, value, unused_deterministic=None):
                for element in value:
                    write(tag_bytes)
                    try:
                        write(local_struct_pack(format, element))
                    except SystemError:
                        EncodeNonFiniteOrRaise(write, element)
            return EncodeRepeatedField
        else:
            tag_bytes = TagBytes(field_number, wire_type)

            def EncodeField(write, value, unused_deterministic=None):
                write(tag_bytes)
                try:
                    write(local_struct_pack(format, value))
                except SystemError:
                    EncodeNonFiniteOrRaise(write, value)
            return EncodeField
    return SpecificEncoder