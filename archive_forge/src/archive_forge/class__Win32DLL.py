import os
from .dependencies import ctypes
class _Win32DLL(object):
    """Helper class to manage the interface with the Win32 runtime"""

    def __init__(self, name):
        self._libname = name
        if name is None:
            self._loaded = False
        else:
            self._loaded = None
        self.dll = None

    def available(self):
        if self._loaded is not None:
            return self._loaded
        self._loaded, self.dll = _load_dll(self._libname)
        if not self._loaded:
            return self._loaded
        self.putenv_s = self.dll.SetEnvironmentVariableA
        self.putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.putenv_s.restype = ctypes.c_bool
        self.wputenv_s = self.dll.SetEnvironmentVariableW
        self.wputenv_s.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        self.wputenv_s.restype = ctypes.c_bool
        self._getenv_dll = self.dll.GetEnvironmentVariableA
        self._getenv_dll.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong]
        self._getenv_dll.restype = ctypes.c_ulong
        self._wgetenv_dll = self.dll.GetEnvironmentVariableW
        self._wgetenv_dll.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong]
        self._wgetenv_dll.restype = ctypes.c_ulong
        self._envstr = self.dll.GetEnvironmentStringsW
        self._envstr.argtypes = []
        self._envstr.restype = ctypes.POINTER(ctypes.c_wchar)
        self._free_envstr = self.dll.FreeEnvironmentStringsW
        self._free_envstr.argtypes = [ctypes.POINTER(ctypes.c_wchar)]
        self._free_envstr.restype = ctypes.c_bool
        return self._loaded

    def getenv(self, key):
        size = self._getenv_dll(key, None, 0)
        if not size:
            return None
        buf = ctypes.create_string_buffer(b'\x00' * size)
        self._getenv_dll(key, buf, size)
        return buf.value or None

    def wgetenv(self, key):
        size = self._wgetenv_dll(key, None, 0)
        if not size:
            return None
        buf = ctypes.create_unicode_buffer(u'\x00' * size)
        self._wgetenv_dll(key, buf, size)
        return buf.value or None

    def get_env_dict(self):
        ans = {}
        _str_buf = self._envstr()
        _null = {u'\x00', b'\x00'}
        i = 0
        while _str_buf[i] not in _null:
            _str = ''
            while _str_buf[i] not in _null:
                _str += _str_buf[i]
                i += len(_str_buf[i])
                if len(_str_buf[i]) == 0:
                    raise ValueError('Error processing Win32 GetEnvironmentStringsW: 0-length character encountered')
                if i > 32767:
                    raise ValueError('Error processing Win32 GetEnvironmentStringsW: exceeded max environment block size (32767)')
            key, val = _str.split('=', 1)
            ans[key] = val
            i += len(_str_buf[i])
        self._free_envstr(_str_buf)
        return ans