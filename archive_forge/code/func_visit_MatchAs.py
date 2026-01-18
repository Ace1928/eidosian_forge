import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_MatchAs(self, node):
    name = node.name
    pattern = node.pattern
    if name is None:
        self.write('_')
    elif pattern is None:
        self.write(node.name)
    else:
        with self.require_parens(_Precedence.TEST, node):
            self.set_precedence(_Precedence.BOR, node.pattern)
            self.traverse(node.pattern)
            self.write(f' as {node.name}')