import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _set_3floats(self, key, values):
    x, y, z = map(float, values)
    al.alListener3f(key, x, y, z)
    self._check_error('Failed to set value.')