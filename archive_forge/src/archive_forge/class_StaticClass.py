import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
class StaticClass(object):

    def __new__(cls, *argv, **argd):
        """Don't try to instance this class, just use the static methods."""
        raise NotImplementedError('Cannot instance static class %s' % cls.__name__)