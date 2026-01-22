from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
class _BlocksIterator(_ValueIterator):
    kind = 'block'

    def _dispose(self):
        self._capi.LLVMPY_DisposeBlocksIter(self)

    def _next(self):
        return ffi.lib.LLVMPY_BlocksIterNext(self)