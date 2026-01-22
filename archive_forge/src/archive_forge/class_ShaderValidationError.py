import logging
from OpenGL import GL
from OpenGL.GL.ARB import (
from OpenGL.extensions import alternate
from OpenGL._bytes import bytes,unicode,as_8_bit
class ShaderValidationError(RuntimeError):
    """Raised when a program fails to validate"""