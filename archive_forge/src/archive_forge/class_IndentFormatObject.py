import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class IndentFormatObject(FormatObject):

    def __init__(self, indent, child):
        assert isinstance(child, FormatObject)
        self.indent = indent
        self.child = child

    def children(self):
        return [self.child]

    def as_tuple(self):
        return ('indent', self.indent, self.child.as_tuple())

    def space_upto_nl(self):
        return self.child.space_upto_nl()

    def flat(self):
        return indent(self.indent, self.child.flat())

    def is_indent(self):
        return True