from OpenGL.latebind import LateBind
from OpenGL._bytes import bytes,unicode,as_8_bit
import OpenGL as root
import sys
import logging
class _Alternate(LateBind):

    def __init__(self, name, *alternates):
        """Initialize set of alternative implementations of the same function"""
        self.__name__ = name
        self._alternatives = alternates
        if root.MODULE_ANNOTATIONS:
            frame = sys._getframe().f_back
            if frame and frame.f_back and ('__name__' in frame.f_back.f_globals):
                self.__module__ = frame.f_back.f_globals['__name__']

    def __bool__(self):
        from OpenGL import error
        try:
            return bool(self.getFinalCall())
        except error.NullFunctionError as err:
            return False
    __nonzero__ = __bool__

    def finalise(self):
        """Call, doing a late lookup and bind to find an implementation"""
        for alternate in self._alternatives:
            if alternate:
                return alternate
        from OpenGL import error
        raise error.NullFunctionError('Attempt to call an undefined alternate function (%s), check for bool(%s) before calling' % (', '.join([x.__name__ for x in self._alternatives]), self.__name__))