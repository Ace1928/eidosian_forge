import struct
from llvmlite.ir._utils import _StrCaching
def _format_double(value):
    """
    Format *value* as a hexadecimal string of its IEEE double precision
    representation.
    """
    return _format_float_as_hex(value, 'd', 'Q', 16)