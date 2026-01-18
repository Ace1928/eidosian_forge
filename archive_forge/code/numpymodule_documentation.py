import logging
from OpenGL import _configflags
import OpenGL
import ctypes
from OpenGL._bytes import long
from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL import error
from OpenGL.arrays import formathandler
from OpenGL import acceleratesupport
Determine dimensions of the passed array value (if possible)