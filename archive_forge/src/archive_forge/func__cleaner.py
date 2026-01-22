from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
def _cleaner(vbo):
    """Construct a mapped-array cleaner function to unmap vbo.target"""

    def clean(ref):
        try:
            _cleaners.pop(vbo)
        except Exception as err:
            pass
        else:
            vbo.implementation.glUnmapBuffer(vbo.target)
    return clean