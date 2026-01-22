import struct
from llvmlite.ir._utils import _StrCaching
class HalfType(_BaseFloatType):
    """
    The type for single-precision floats.
    """
    null = '0.0'
    intrinsic_name = 'f16'

    def __str__(self):
        return 'half'

    def format_constant(self, value):
        return _format_double(_as_half(value))