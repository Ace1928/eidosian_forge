import ctypes, logging, os, sys
from ctypes import util
import OpenGL
def _loadLibraryPosix(dllType, name, mode):
    """Load a given library for posix systems

    The problem with util.find_library is that it does not respect linker runtime variables like
    LD_LIBRARY_PATH.

    Also we cannot rely on libGLU.so to be available, for example. Most of Linux distributions will
    ship only libGLU.so.1 by default. Files ending with .so are normally used when compiling and are
    provided by dev packages.

    returns the ctypes C-module object
    """
    prefix = 'lib'
    suffix = '.so'
    base_name = prefix + name + suffix
    filenames_to_try = [base_name]
    filenames_to_try.extend(list(reversed([base_name + '.%i' % i for i in range(0, 10)])))
    err = None
    for filename in filenames_to_try:
        try:
            result = dllType(filename, mode)
            _log.debug('Loaded %s => %s %s', base_name, filename, result)
            return result
        except Exception as current_err:
            err = current_err
    _log.info('Failed to load library ( %r ): %s', filename, err or 'No filenames available to guess?')