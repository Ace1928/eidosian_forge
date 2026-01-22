import struct
from llvmlite.ir._utils import _StrCaching
def _as_float(value):
    """
    Truncate to single-precision float.
    """
    return struct.unpack('f', struct.pack('f', value))[0]