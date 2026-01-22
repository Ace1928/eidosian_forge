from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
def _base_glutInit(pargc, argv):
    """Overrides base glut init with exit-function-aware version"""
    return __glutInitWithExit(pargc, argv, _exitfunc)