import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
class SynchronizedString(SynchronizedArray):
    value = make_property('value')
    raw = make_property('raw')