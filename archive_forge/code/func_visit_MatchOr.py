import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_MatchOr(self, node):
    with self.require_parens(_Precedence.BOR, node):
        self.set_precedence(_Precedence.BOR.next(), *node.patterns)
        self.interleave(lambda: self.write(' | '), self.traverse, node.patterns)