import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_MatchClass(self, node):
    self.set_precedence(_Precedence.ATOM, node.cls)
    self.traverse(node.cls)
    with self.delimit('(', ')'):
        patterns = node.patterns
        self.interleave(lambda: self.write(', '), self.traverse, patterns)
        attrs = node.kwd_attrs
        if attrs:

            def write_attr_pattern(pair):
                attr, pattern = pair
                self.write(f'{attr}=')
                self.traverse(pattern)
            if patterns:
                self.write(', ')
            self.interleave(lambda: self.write(', '), write_attr_pattern, zip(attrs, node.kwd_patterns, strict=True))