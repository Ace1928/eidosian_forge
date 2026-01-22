import ctypes, logging, os, sys
from ctypes import util
import OpenGL
def _loadLibraryWindows(dllType, name, mode):
    """Load a given library for Windows systems

    returns the ctypes C-module object
    """
    fullName = None
    try:
        fullName = util.find_library(name)
        if fullName is not None:
            name = fullName
        elif os.path.isfile(os.path.join(DLL_DIRECTORY, name + '.dll')):
            name = os.path.join(DLL_DIRECTORY, name + '.dll')
    except Exception as err:
        _log.info('Failed on util.find_library( %r ): %s', name, err)
        pass
    try:
        return dllType(name, mode)
    except Exception as err:
        err.args += (name, fullName)
        raise