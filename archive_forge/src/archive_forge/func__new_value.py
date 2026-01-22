import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
def _new_value(type_):
    size = ctypes.sizeof(type_)
    wrapper = heap.BufferWrapper(size)
    return rebuild_ctype(type_, wrapper, None)