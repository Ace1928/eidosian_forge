from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
def as_bitcode(self):
    """
        Return the module's LLVM bitcode, as a bytes object.
        """
    ptr = c_char_p(None)
    size = c_size_t(-1)
    ffi.lib.LLVMPY_WriteBitcodeToString(self, byref(ptr), byref(size))
    if not ptr:
        raise MemoryError
    try:
        assert size.value >= 0
        return string_at(ptr, size.value)
    finally:
        ffi.lib.LLVMPY_DisposeString(ptr)