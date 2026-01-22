import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
@functools.wraps(func)
def _with_np_array(*args, **kwargs):
    with np_array(active=True):
        return func(*args, **kwargs)