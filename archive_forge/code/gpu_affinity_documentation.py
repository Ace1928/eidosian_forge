from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.NV.gpu_affinity import *
from OpenGL.raw.WGL.NV.gpu_affinity import _EXTENSION_NAME
Return boolean indicating whether this extension is available