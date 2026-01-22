import sys
import ctypes
from pyglet.util import debug_print
class VOIDMETHOD(METHOD):
    """COM method with no return value."""

    def __init__(self, *args):
        super().__init__(None, *args)