from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class SourceRange(Structure):
    """
    A SourceRange describes a range of source locations within the source
    code.
    """
    _fields_ = [('ptr_data', c_void_p * 2), ('begin_int_data', c_uint), ('end_int_data', c_uint)]

    @staticmethod
    def from_locations(start, end):
        return conf.lib.clang_getRange(start, end)

    @property
    def start(self):
        """
        Return a SourceLocation representing the first character within a
        source range.
        """
        return conf.lib.clang_getRangeStart(self)

    @property
    def end(self):
        """
        Return a SourceLocation representing the last character within a
        source range.
        """
        return conf.lib.clang_getRangeEnd(self)

    def __eq__(self, other):
        return conf.lib.clang_equalRanges(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, other):
        """Useful to detect the Token/Lexer bug"""
        if not isinstance(other, SourceLocation):
            return False
        if other.file is None and self.start.file is None:
            pass
        elif self.start.file.name != other.file.name or other.file.name != self.end.file.name:
            return False
        if self.start.line < other.line < self.end.line:
            return True
        elif self.start.line == other.line:
            if self.start.column <= other.column:
                return True
        elif other.line == self.end.line:
            if other.column <= self.end.column:
                return True
        return False

    def __repr__(self):
        return '<SourceRange start %r, end %r>' % (self.start, self.end)