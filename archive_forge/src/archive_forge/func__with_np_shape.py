import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
@functools.wraps(func)
def _with_np_shape(*args, **kwargs):
    with np_shape(active=True):
        return func(*args, **kwargs)