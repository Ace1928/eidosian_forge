import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
def unregister_lock_callback(acq_fn, rel_fn):
    """Remove the registered callback functions for lock acquire and release.
    The arguments are the same as used in `register_lock_callback()`.
    """
    lib._lock.unregister(acq_fn, rel_fn)