import enum
import struct
import typing
def _unpack_value(b_mem: memoryview, offset: int) -> typing.Tuple[bytes, int]:
    """Unpacks a raw C struct value to a byte string."""
    length = struct.unpack('<I', b_mem[offset:offset + 4].tobytes())[0]
    new_offset = offset + length + 4
    data = b''
    if length:
        data = b_mem[offset + 4:offset + 4 + length].tobytes()
    return (data, new_offset)