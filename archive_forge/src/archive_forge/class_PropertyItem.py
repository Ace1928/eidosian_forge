from pyglet.libs.win32.com import pIUnknown
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
class PropertyItem(Structure):
    _fields_ = [('id', c_uint), ('length', c_ulong), ('type', c_short), ('value', c_void_p)]