import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
@argtypes.setter
def argtypes(self, argtypes):
    self._cfn.argtypes = argtypes