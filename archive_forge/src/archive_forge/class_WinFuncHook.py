import ctypes
import functools
from winappdbg import compat
import sys
class WinFuncHook(object):

    def __init__(self, name):
        self.__name = name

    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattr__(self, name)
        return WinCallHook(self.__name, name)