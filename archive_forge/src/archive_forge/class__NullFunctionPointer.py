import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
class _NullFunctionPointer(object):
    """Function-pointer-like object for undefined functions"""

    def __init__(self, name, dll, resultType, argTypes, argNames, extension=None, doc=None, deprecated=False, error_checker=None, force_extension=None):
        from OpenGL import error
        self.__name__ = name
        self.DLL = dll
        self.argNames = argNames
        self.argtypes = argTypes
        self.errcheck = None
        self.restype = resultType
        self.extension = extension
        self.doc = doc
        self.deprecated = deprecated
        self.error_checker = error_checker
        self.force_extension = force_extension
    resolved = False

    def __nonzero__(self):
        """Make this object appear to be NULL"""
        if not self.resolved and (self.extension or self.force_extension):
            self.load()
        return self.resolved
    __bool__ = __nonzero__

    def load(self):
        """Attempt to load the function again, presumably with a context this time"""
        try:
            from OpenGL import platform
        except ImportError:
            if log:
                log.info('Platform import failed (likely during shutdown)')
            return None
        try:
            func = platform.PLATFORM.constructFunction(self.__name__, self.DLL, resultType=self.restype, argTypes=self.argtypes, doc=self.doc, argNames=self.argNames, extension=self.extension, error_checker=self.error_checker, force_extension=self.force_extension)
        except AttributeError as err:
            return None
        else:
            self.__class__.__call__ = staticmethod(func.__call__)
            self.resolved = True
            return func
        return None

    def __call__(self, *args, **named):
        if self.load():
            return self(*args, **named)
        else:
            try:
                from OpenGL import error
            except ImportError as err:
                pass
            else:
                raise error.NullFunctionError('Attempt to call an undefined function %s, check for bool(%s) before calling' % (self.__name__, self.__name__))