import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
class _ObjectCacheRef(ffi.ObjectRef):
    """
    Internal: an ObjectCache instance for use within an ExecutionEngine.
    """

    def __init__(self, obj):
        ptr = ffi.lib.LLVMPY_CreateObjectCache(_notify_c_hook, _getbuffer_c_hook, obj)
        ffi.ObjectRef.__init__(self, ptr)

    def _dispose(self):
        self._capi.LLVMPY_DisposeObjectCache(self)