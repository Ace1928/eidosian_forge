from array import array
from threading import Lock
import traceback
import ctypes
from ctypes import c_int, c_void_p, CFUNCTYPE, POINTER, cast
from .base import _LIB, check_call, string_types, mx_uint
from .base import NDArrayHandle, c_array, c_handle_array, c_array_buf, MXCallbackList, SymbolHandle
from .ndarray import NDArray, _ndarray_cls
from .ndarray import _GRAD_REQ_MAP
from .symbol import Symbol
def train_mode():
    """Returns a scope context to be used in 'with' statement
    in which forward pass behavior is set to training mode,
    without changing the recording states.

    Example::

        y = model(x)
        with autograd.train_mode():
            y = dropout(y)

    """
    return _RecordingStateScope(None, True)