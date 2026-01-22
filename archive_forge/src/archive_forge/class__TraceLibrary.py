import os
import re
import sys
import ctypes
import ctypes.util
import pyglet
class _TraceLibrary:

    def __init__(self, library):
        self._library = library
        print(library)

    def __getattr__(self, name):
        func = getattr(self._library, name)
        f = _TraceFunction(func)
        return f