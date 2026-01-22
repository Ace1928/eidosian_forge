from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def _load_python_helper_lib():
    try:
        return _load_python_helper_lib.__lib__
    except AttributeError:
        pass
    with _lock:
        try:
            return _load_python_helper_lib.__lib__
        except AttributeError:
            pass
        lib = _load_python_helper_lib_uncached()
        _load_python_helper_lib.__lib__ = lib
        return lib