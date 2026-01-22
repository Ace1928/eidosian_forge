import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def add_ir(self, llvmir):
    """
        Adds a compilation unit to the library using LLVM IR as the input
        format.

        This takes a string or an object that can be converted to a string,
        including IRBuilder, that contains LLVM IR.
        """
    self.__entries.append((0, str(llvmir).encode('utf-8')))
    return self