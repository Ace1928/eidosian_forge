import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def wrapContextCheck(self, func, dll):
    """Wrap function with context-checking if appropriate"""
    if _configflags.CONTEXT_CHECKING and dll is self.GL and (func.__name__ not in ('glGetString', 'glGetStringi', 'glGetIntegerv')) and (not func.__name__.startswith('glX')):
        return _CheckContext(func, self.CurrentContextIsValid)
    return func