import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _for_helper(self, fill, node):
    self.fill(fill)
    self.set_precedence(_Precedence.TUPLE, node.target)
    self.traverse(node.target)
    self.write(' in ')
    self.traverse(node.iter)
    with self.block(extra=self.get_type_comment(node)):
        self.traverse(node.body)
    if node.orelse:
        self.fill('else')
        with self.block():
            self.traverse(node.orelse)