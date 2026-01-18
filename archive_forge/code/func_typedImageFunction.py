from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
def typedImageFunction(suffix, arrayConstant, baseFunction):
    """Produce a typed version of the given image function"""
    functionName = baseFunction.__name__
    functionName = '%(functionName)s%(suffix)s' % locals()
    arrayType = arrays.GL_CONSTANT_TO_ARRAY_TYPE[arrayConstant]
    function = setDimensionsAsInts(setImageInput(baseFunction, arrayType, typeName=arrayConstant))
    return (functionName, function)