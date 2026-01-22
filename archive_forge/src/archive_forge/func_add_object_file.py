import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def add_object_file(self, file_path):
    """
        Adds a compilation unit to the library using pre-compiled object file.

        This takes a string or path-like object that references an object file
        which will be loaded by LLVM.
        """
    with open(file_path, 'rb') as f:
        self.__entries.append((2, f.read()))
    return self