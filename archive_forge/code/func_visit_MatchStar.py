import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_MatchStar(self, node):
    name = node.name
    if name is None:
        name = '_'
    self.write(f'*{name}')