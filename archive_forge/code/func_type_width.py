from ctypes import c_int, c_bool, c_void_p, c_uint64
import enum
from llvmlite.binding import ffi
@property
def type_width(self):
    """
        Return the basic size of this type if it is a primitive type. These are
        fixed by LLVM and are not target-dependent.
        This will return zero if the type does not have a size or is not a
        primitive type.

        If this is a scalable vector type, the scalable property will be set and
        the runtime size will be a positive integer multiple of the base size.

        Note that this may not reflect the size of memory allocated for an
        instance of the type or the number of bytes that are written when an
        instance of the type is stored to memory.
        """
    return ffi.lib.LLVMPY_GetTypeBitWidth(self)