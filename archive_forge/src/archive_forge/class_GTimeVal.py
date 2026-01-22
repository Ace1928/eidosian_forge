from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
class GTimeVal(Structure):
    _fields_ = [('tv_sec', c_long), ('tv_usec', c_long)]