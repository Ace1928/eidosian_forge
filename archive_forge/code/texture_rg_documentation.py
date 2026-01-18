from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.EXT.texture_rg import *
from OpenGL.raw.GLES2.EXT.texture_rg import _EXTENSION_NAME
from OpenGL import images as _images
Return boolean indicating whether this extension is available