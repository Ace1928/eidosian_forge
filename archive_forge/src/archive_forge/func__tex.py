from OpenGL.raw import GLU as _simple
from OpenGL import platform, converters, wrapper
from OpenGL.GLU import glustruct
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import arrays, error
import ctypes
import weakref
from OpenGL.platform import PLATFORM
import OpenGL
from OpenGL import _configflags
def _tex(self, function):
    """Texture coordinate callback

        NOTE: there is no way for *us* to tell what size the array is, you will
        get back a raw data-point, not an array, as you do for all other callback
        types!!!
        """

    def oor(*args):
        if len(args) > 1:
            oor = self.originalObject(args[1])
            return function(args[0], oor)
        else:
            return function(args[0])
    return oor