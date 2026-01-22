import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def _raw_object_cache_notify(self, data):
    """
        Low-level notify hook.
        """
    if self._object_cache_notify is None:
        return
    module_ptr = data.contents.module_ptr
    buf_ptr = data.contents.buf_ptr
    buf_len = data.contents.buf_len
    buf = string_at(buf_ptr, buf_len)
    module = self._find_module_ptr(module_ptr)
    if module is None:
        raise RuntimeError('object compilation notification for unknown module %s' % (module_ptr,))
    self._object_cache_notify(module, buf)