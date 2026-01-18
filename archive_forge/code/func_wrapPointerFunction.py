from OpenGL import platform, error, wrapper, contextdata, converters, constant
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
import ctypes
def wrapPointerFunction(name, baseFunction, glType, arrayType, startArgs, defaultSize):
    """Wrap the given pointer-setting function"""
    function = wrapper.wrapper(baseFunction)
    if 'ptr' in baseFunction.argNames:
        pointer_name = 'ptr'
    else:
        pointer_name = 'pointer'
    assert not getattr(function, 'pyConverters', None), 'Reusing wrappers?'
    if arrayType:
        arrayModuleType = arraydatatype.GL_CONSTANT_TO_ARRAY_TYPE[glType]
        function.setPyConverter(pointer_name, arrayhelpers.asArrayType(arrayModuleType))
    else:
        function.setPyConverter(pointer_name, arrayhelpers.AsArrayOfType(pointer_name, 'type'))
    function.setCConverter(pointer_name, converters.getPyArgsName(pointer_name))
    if 'size' in function.argNames:
        function.setPyConverter('size')
        function.setCConverter('size', arrayhelpers.arraySizeOfFirstType(arrayModuleType, defaultSize))
    if 'type' in function.argNames:
        function.setPyConverter('type')
        function.setCConverter('type', glType)
    if 'stride' in function.argNames:
        function.setPyConverter('stride')
        function.setCConverter('stride', 0)
    function.setStoreValues(arrayhelpers.storePointerType(pointer_name, arrayType))
    function.setReturnValues(wrapper.returnPyArgument(pointer_name))
    return (name, function)