import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class NAryFormatObject(FormatObject):

    def __init__(self, fs):
        assert all([isinstance(a, FormatObject) for a in fs])
        self.children = fs

    def children(self):
        return self.children