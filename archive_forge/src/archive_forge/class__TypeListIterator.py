from ctypes import c_int, c_bool, c_void_p, c_uint64
import enum
from llvmlite.binding import ffi
class _TypeListIterator(_TypeIterator):

    def _dispose(self):
        self._capi.LLVMPY_DisposeElementIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_ElementIterNext(self)