import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_FunctionType(self, node):
    with self.delimit('(', ')'):
        self.interleave(lambda: self.write(', '), self.traverse, node.argtypes)
    self.write(' -> ')
    self.traverse(node.returns)