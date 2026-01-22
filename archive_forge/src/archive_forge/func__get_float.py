import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def _get_float(self, key):
    al_float = al.ALfloat()
    al.alGetListenerf(key, al_float)
    self._check_error('Failed to get value')
    return al_float.value