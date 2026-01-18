import ctypes, logging
from OpenGL import platform, error
from OpenGL._configflags import STORE_POINTERS, ERROR_ON_COPY, SIZE_1_ARRAY_UNPACK
from OpenGL import converters
from OpenGL.converters import DefaultCConverter
from OpenGL.converters import returnCArgument,returnPyArgument
from OpenGL.latebind import LateBind
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def wrapperCall(*args):
    """Wrapper with all save returnValues and storeValues"""
    cArguments = args
    try:
        result = wrappedOperation(*cArguments)
    except ctypes.ArgumentError as err:
        err.args = err.args + (cArguments,)
        raise err
    except error.GLError as err:
        err.cArgs = cArguments
        err.pyArgs = args
        raise err
    return result