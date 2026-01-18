import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def write_attr_pattern(pair):
    attr, pattern = pair
    self.write(f'{attr}=')
    self.traverse(pattern)