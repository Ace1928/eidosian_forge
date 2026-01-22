import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class StringLengths(CConverter):
    """CConverter for processing array-of-pointers-to-strings data-type

    Converter is a CConverter for the array-of-lengths for a
    array-of-pointers-to-strings data-type used to pass a set
    of code fragments to the GLSL compiler.

    Provides also:

        stringArray -- PyConverter callable ensuring list-of-strings
            format for the python argument

        stringArrayForC -- CResolver converting the array to
            POINTER(c_char_p) format for passing to C

        totalCount -- CConverter callable giving count of string
            pointers (that is, length of the pointer array)
    """
    argNames = ('name',)
    indexLookups = [('index', 'name', 'pyArgIndex')]
    __slots__ = ()

    def __call__(self, pyArgs, index, baseOperation):
        """Get array of length integers for string contents"""
        from OpenGL.raw.GL import _types
        tmp = [len(x) for x in pyArgs[self.index]]
        a_type = _types.GLint * len(tmp)
        return a_type(*tmp)

    def totalCount(self, pyArgs, index, baseOperation):
        """Get array of length integers for string contents"""
        return len(pyArgs[self.index])

    def stringArray(self, arg, baseOperation, args):
        """Create basic array-of-strings object from pyArg"""
        if isinstance(arg, (bytes, unicode)):
            arg = [arg]
        value = [as_8_bit(x) for x in arg]
        return value

    def stringArrayForC(self, strings):
        """Create a ctypes pointer to char-pointer set"""
        from OpenGL import arrays
        result = (ctypes.c_char_p * len(strings))()
        for i, s in enumerate(strings):
            result[i] = ctypes.cast(arrays.GLcharARBArray.dataPointer(s), ctypes.c_char_p)
        return result