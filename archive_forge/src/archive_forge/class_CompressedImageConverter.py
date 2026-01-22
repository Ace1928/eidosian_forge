from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
class CompressedImageConverter(object):

    def finalise(self, wrapper):
        """Get our pixel index from the wrapper"""
        self.dataIndex = wrapper.pyArgIndex('data')

    def __call__(self, pyArgs, index, wrappedOperation):
        """Create a data-size measurement for our image"""
        arg = pyArgs[self.dataIndex]
        return arraydatatype.ArrayDatatype.arrayByteCount(arg)