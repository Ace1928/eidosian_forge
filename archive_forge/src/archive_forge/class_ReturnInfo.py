import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
class ReturnInfo(object):

    def __init__(self, return_line):
        self.return_line = return_line

    def __str__(self):
        return '{return: %s}' % (self.return_line,)
    __repr__ = __str__