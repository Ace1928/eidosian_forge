import os
from .dependencies import ctypes
def wgetenv(self, key):
    size = self._wgetenv_dll(key, None, 0)
    if not size:
        return None
    buf = ctypes.create_unicode_buffer(u'\x00' * size)
    self._wgetenv_dll(key, buf, size)
    return buf.value or None