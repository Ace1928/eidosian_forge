import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
FreeType typographic face object.

    Keeps the reference count to the face at +1 as long as this object exists. If other objects
    want to keep a face without a reference to this object, they should increase the reference
    counter themselves and decrease it again when done.
    