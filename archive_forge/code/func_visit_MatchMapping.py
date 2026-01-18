import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_MatchMapping(self, node):

    def write_key_pattern_pair(pair):
        k, p = pair
        self.traverse(k)
        self.write(': ')
        self.traverse(p)
    with self.delimit('{', '}'):
        keys = node.keys
        self.interleave(lambda: self.write(', '), write_key_pattern_pair, zip(keys, node.patterns, strict=True))
        rest = node.rest
        if rest is not None:
            if keys:
                self.write(', ')
            self.write(f'**{rest}')