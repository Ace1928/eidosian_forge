from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
class _ValueIterator(ffi.ObjectRef):
    kind = None

    def __init__(self, ptr, parents):
        ffi.ObjectRef.__init__(self, ptr)
        self._parents = parents
        if self.kind is None:
            raise NotImplementedError('%s must specify kind attribute' % (type(self).__name__,))

    def __next__(self):
        vp = self._next()
        if vp:
            return ValueRef(vp, self.kind, self._parents)
        else:
            raise StopIteration
    next = __next__

    def __iter__(self):
        return self