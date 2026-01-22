import struct
import zlib
from .static_tuple import StaticTuple
def _search_key_255(key):
    """Map the key tuple into a search key string which has 255-way fan out.

    We use 255-way because '
' is used as a delimiter, and causes problems
    while parsing.
    """
    data = b'\x00'.join([struct.pack('>L', _crc32(bit)) for bit in key])
    return data.replace(b'\n', b'_')