from OpenGL.raw.GLUT.constants import *
from ctypes import *
from OpenGL._bytes import unicode
from OpenGL import platform, arrays
from OpenGL.constant import Constant
from OpenGL.raw.GL import _types as GL_types
from OpenGL.raw.GL._types import (
class STRING(c_char_p):

    @classmethod
    def from_param(cls, value):
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        return c_char_p.from_param(value)