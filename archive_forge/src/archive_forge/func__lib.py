import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
@property
def _lib(self):
    if not self._lib_handle:
        self._load_lib()
    return self._lib_handle