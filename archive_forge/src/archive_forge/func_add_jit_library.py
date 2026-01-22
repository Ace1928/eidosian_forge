import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def add_jit_library(self, name):
    """
        Adds an existing JIT library as prerequisite.

        The name of the library must match the one provided in a previous link
        command.
        """
    self.__entries.append((3, str(name).encode('utf-8')))
    return self