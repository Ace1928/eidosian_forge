import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
class TryExceptInfo(object):

    def __init__(self, try_line, ignore=False):
        """
        :param try_line:
        :param ignore:
            Usually we should ignore any block that's not a try..except
            (this can happen for finally blocks, with statements, etc, for
            which we create temporary entries).
        """
        self.try_line = try_line
        self.ignore = ignore
        self.except_line = -1
        self.except_end_line = -1
        self.raise_lines_in_except = []
        self.except_bytecode_offset = -1
        self.except_end_bytecode_offset = -1

    def is_line_in_try_block(self, line):
        return self.try_line <= line < self.except_line

    def is_line_in_except_block(self, line):
        return self.except_line <= line <= self.except_end_line

    def __str__(self):
        lst = ['{try:', str(self.try_line), ' except ', str(self.except_line), ' end block ', str(self.except_end_line)]
        if self.raise_lines_in_except:
            lst.append(' raises: %s' % (', '.join((str(x) for x in self.raise_lines_in_except)),))
        lst.append('}')
        return ''.join(lst)
    __repr__ = __str__