from ctypes import c_int, c_bool, c_void_p, c_uint64
import enum
from llvmlite.binding import ffi
class _TypeIterator(ffi.ObjectRef):

    def __next__(self):
        vp = self._next()
        if vp:
            return TypeRef(vp)
        else:
            raise StopIteration
    next = __next__

    def __iter__(self):
        return self