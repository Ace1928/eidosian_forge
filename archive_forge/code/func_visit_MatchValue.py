import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_MatchValue(self, node):
    self.traverse(node.value)