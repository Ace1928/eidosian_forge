import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def write_key_pattern_pair(pair):
    k, p = pair
    self.traverse(k)
    self.write(': ')
    self.traverse(p)