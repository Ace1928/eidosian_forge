import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning

    Return a synchronization wrapper for a RawArray
    