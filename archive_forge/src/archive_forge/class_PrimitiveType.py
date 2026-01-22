import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class PrimitiveType(BasePrimitiveType):
    _attrs_ = ('name',)
    ALL_PRIMITIVE_TYPES = {'char': 'c', 'short': 'i', 'int': 'i', 'long': 'i', 'long long': 'i', 'signed char': 'i', 'unsigned char': 'i', 'unsigned short': 'i', 'unsigned int': 'i', 'unsigned long': 'i', 'unsigned long long': 'i', 'float': 'f', 'double': 'f', 'long double': 'f', 'float _Complex': 'j', 'double _Complex': 'j', '_Bool': 'i', 'wchar_t': 'c', 'char16_t': 'c', 'char32_t': 'c', 'int8_t': 'i', 'uint8_t': 'i', 'int16_t': 'i', 'uint16_t': 'i', 'int32_t': 'i', 'uint32_t': 'i', 'int64_t': 'i', 'uint64_t': 'i', 'int_least8_t': 'i', 'uint_least8_t': 'i', 'int_least16_t': 'i', 'uint_least16_t': 'i', 'int_least32_t': 'i', 'uint_least32_t': 'i', 'int_least64_t': 'i', 'uint_least64_t': 'i', 'int_fast8_t': 'i', 'uint_fast8_t': 'i', 'int_fast16_t': 'i', 'uint_fast16_t': 'i', 'int_fast32_t': 'i', 'uint_fast32_t': 'i', 'int_fast64_t': 'i', 'uint_fast64_t': 'i', 'intptr_t': 'i', 'uintptr_t': 'i', 'intmax_t': 'i', 'uintmax_t': 'i', 'ptrdiff_t': 'i', 'size_t': 'i', 'ssize_t': 'i'}

    def __init__(self, name):
        assert name in self.ALL_PRIMITIVE_TYPES
        self.name = name
        self.c_name_with_marker = name + '&'

    def is_char_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'c'

    def is_integer_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'i'

    def is_float_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'f'

    def is_complex_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'j'

    def build_backend_type(self, ffi, finishlist):
        return global_cache(self, ffi, 'new_primitive_type', self.name)