from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
class TypedImageInputConverter(ImageInputConverter):

    def __init__(self, rank, pixelsName, arrayType, typeName=None):
        self.rank = rank
        self.arrayType = arrayType
        self.pixelsName = pixelsName
        self.typeName = typeName

    def __call__(self, arg, baseOperation, pyArgs):
        """The pyConverter for the pixels"""
        images.setupDefaultTransferMode()
        images.rankPacking(self.rank)
        return self.arrayType.asArray(arg)

    def finalise(self, wrapper):
        """Get our pixel index from the wrapper"""
        self.pixelsIndex = wrapper.pyArgIndex(self.pixelsName)

    def width(self, pyArgs, index, wrappedOperation):
        """Extract the width from the pixels argument"""
        return self.arrayType.dimensions(pyArgs[self.pixelsIndex])[0]

    def height(self, pyArgs, index, wrappedOperation):
        """Extract the height from the pixels argument"""
        return self.arrayType.dimensions(pyArgs[self.pixelsIndex])[1]

    def depth(self, pyArgs, index, wrappedOperation):
        """Extract the depth from the pixels argument"""
        return self.arrayType.dimensions(pyArgs[self.pixelsIndex])[2]

    def type(self, pyArgs, index, wrappedOperation):
        """Provide the item-type argument from our stored value

        This is used for pre-bound processing where we want to provide
        the type by implication...
        """
        return self.typeName