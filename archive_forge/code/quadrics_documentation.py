from OpenGL.raw import GLU as _simple
from OpenGL.platform import createBaseFunction, PLATFORM
import ctypes
Register a callback for the quadric object
        
        At the moment only GLU_ERROR is supported by OpenGL, but
        we allow for the possibility of more callbacks in the future...
        