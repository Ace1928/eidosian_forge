import os
from .dependencies import ctypes
def wputenv_s(self, key, val):
    if not val:
        if key in os.environ:
            del os.environ[key]
        return
    os.environ[key] = val