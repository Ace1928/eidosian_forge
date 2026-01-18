import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def wrapLogging(self, func):
    """Wrap function with logging operations if appropriate"""
    return logs.logOnFail(func, logs.getLog('OpenGL.errors'))