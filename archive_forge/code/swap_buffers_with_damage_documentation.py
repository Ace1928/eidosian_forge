from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.EXT.swap_buffers_with_damage import *
from OpenGL.raw.EGL.EXT.swap_buffers_with_damage import _EXTENSION_NAME
Return boolean indicating whether this extension is available