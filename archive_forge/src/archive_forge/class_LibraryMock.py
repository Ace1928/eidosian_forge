import os
import re
import sys
import ctypes
import ctypes.util
import pyglet
class LibraryMock:
    """Mock library used when generating documentation."""

    def __getattr__(self, name):
        return LibraryMock()

    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return LibraryMock()

    def __rshift__(self, other):
        return 0