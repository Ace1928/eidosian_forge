import struct
from llvmlite.ir._utils import _StrCaching
@classmethod
def __new(cls, bits):
    assert isinstance(bits, int) and bits >= 0
    self = super(IntType, cls).__new__(cls)
    self.width = bits
    return self