from ctypes import c_void_p, c_char_p, c_bool, POINTER
from llvmlite.binding import ffi
from llvmlite.binding.common import _encode_string
def address_of_symbol(name):
    """
    Get the in-process address of symbol named *name*.
    An integer is returned, or None if the symbol isn't found.
    """
    return ffi.lib.LLVMPY_SearchAddressOfSymbol(_encode_string(name))