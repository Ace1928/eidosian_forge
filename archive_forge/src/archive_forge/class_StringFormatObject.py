import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class StringFormatObject(FormatObject):

    def __init__(self, string):
        assert isinstance(string, str)
        self.string = string

    def is_string(self):
        return True

    def as_tuple(self):
        return self.string

    def space_upto_nl(self):
        return (getattr(self, 'size', len(self.string)), False)