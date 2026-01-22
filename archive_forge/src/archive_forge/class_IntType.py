import struct
from llvmlite.ir._utils import _StrCaching
class IntType(Type):
    """
    The type for integers.
    """
    null = '0'
    _instance_cache = {}
    width: int

    def __new__(cls, bits):
        if 0 <= bits <= 128:
            try:
                return cls._instance_cache[bits]
            except KeyError:
                inst = cls._instance_cache[bits] = cls.__new(bits)
                return inst
        return cls.__new(bits)

    @classmethod
    def __new(cls, bits):
        assert isinstance(bits, int) and bits >= 0
        self = super(IntType, cls).__new__(cls)
        self.width = bits
        return self

    def __getnewargs__(self):
        return (self.width,)

    def __copy__(self):
        return self

    def _to_string(self):
        return 'i%u' % (self.width,)

    def __eq__(self, other):
        if isinstance(other, IntType):
            return self.width == other.width
        else:
            return False

    def __hash__(self):
        return hash(IntType)

    def format_constant(self, val):
        if isinstance(val, bool):
            return str(val).lower()
        else:
            return str(val)

    def wrap_constant_value(self, val):
        if val is None:
            return 0
        return val

    @property
    def intrinsic_name(self):
        return str(self)