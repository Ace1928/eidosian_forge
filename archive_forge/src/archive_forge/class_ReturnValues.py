import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class ReturnValues(Converter):
    """Converter sub-class for use as Wrapper.returnValues

    This class just defines the interface for a returnValues-style
    Converter object
    """

    def __call__(self, result, baseOperation, pyArgs, cArgs):
        """Return a final value to the caller

        result -- the raw ctypes result value
        baseOperation -- the Wrapper object which we are supporting
        pyArgs -- the set of Python arguments produced by pyConverters
        cArgs -- the set of C-compatible arguments produced by CConverter

        return the Python object for the final result
        """
        raise NotImplemented("%s class doesn't implement __call__" % (self.__class__.__name__,))