import ctypes, _ctypes
from OpenGL.raw.GL import _types 
from OpenGL.arrays import _arrayconstants as GL_1_1
from OpenGL import constant
from OpenGL.arrays import formathandler
from OpenGL._bytes import bytes,unicode,as_8_bit
import operator
class CtypesParameterHandler(formathandler.FormatHandler):
    """Ctypes Paramater-type-specific data-type handler for OpenGL"""
    isOutput = True
    HANDLED_TYPES = (ParamaterType, _ctypes._SimpleCData)

    def from_param(cls, value, typeCode=None):
        if isinstance(value, DIRECT_RETURN_TYPES):
            return value
        else:
            return ctypes.byref(value)
    from_param = voidDataPointer = classmethod(from_param)

    def dataPointer(cls, value):
        if isinstance(value, DIRECT_RETURN_TYPES):
            return value
        else:
            return ctypes.addressof(value)
    dataPointer = classmethod(dataPointer)

    def zeros(self, dims, typeCode):
        """Return Numpy array of zeros in given size"""
        type = GL_TYPE_TO_ARRAY_MAPPING[typeCode]
        for dim in dims:
            type *= dim
        return type()

    def ones(self, dims, typeCode='d'):
        """Return numpy array of ones in given size"""
        raise NotImplementedError("Haven't got a good ones implementation yet")

    def arrayToGLType(self, value):
        """Given a value, guess OpenGL type of the corresponding pointer"""
        if isinstance(value, ParamaterType):
            value = value._obj
        result = ARRAY_TO_GL_TYPE_MAPPING.get(value._type_)
        if result is not None:
            return result
        raise TypeError("Don't know GL type for array of type %r, known types: %s\nvalue:%s" % (value._type_, list(ARRAY_TO_GL_TYPE_MAPPING.keys()), value))

    def arraySize(self, value, typeCode=None):
        """Given a data-value, calculate dimensions for the array"""
        if isinstance(value, ParamaterType):
            value = value._obj
        dims = 1
        for base in self.types(value):
            length = getattr(base, '_length_', None)
            if length is not None:
                dims *= length
        return dims

    def arrayByteCount(self, value, typeCode=None):
        """Given a data-value, calculate number of bytes required to represent"""
        if isinstance(value, ParamaterType):
            value = value._obj
        return ctypes.sizeof(value)

    def types(self, value):
        """Produce iterable producing all composite types"""
        if isinstance(value, ParamaterType):
            value = value._obj
        dimObject = value
        while dimObject is not None:
            yield dimObject
            dimObject = getattr(dimObject, '_type_', None)
            if isinstance(dimObject, (bytes, unicode)):
                dimObject = None

    def dims(self, value):
        """Produce iterable of all dimensions"""
        if isinstance(value, ParamaterType):
            value = value._obj
        for base in self.types(value):
            length = getattr(base, '_length_', None)
            if length is not None:
                yield length

    def asArray(self, value, typeCode=None):
        """Convert given value to an array value of given typeCode"""
        if isinstance(value, DIRECT_RETURN_TYPES):
            return value
        if isinstance(value, ParamaterType):
            value = value._obj
        return ctypes.byref(value)

    def unitSize(self, value, typeCode=None):
        """Determine unit size of an array (if possible)"""
        if isinstance(value, ParamaterType):
            value = value._obj
        return tuple(self.dims(value))[-1]

    def dimensions(self, value, typeCode=None):
        """Determine dimensions of the passed array value (if possible)"""
        if isinstance(value, ParamaterType):
            value = value._obj
        return tuple(self.dims(value))