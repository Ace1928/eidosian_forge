from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def LFloat32(name):
    """little endian, 32-bit IEEE floating point number"""
    return FormatField(name, '<', 'f')