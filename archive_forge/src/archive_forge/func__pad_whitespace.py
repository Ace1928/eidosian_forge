import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _pad_whitespace(source):
    """Replace all chars except '\\f\\t' in a line with spaces."""
    result = ''
    for c in source:
        if c in '\x0c\t':
            result += c
        else:
            result += ' '
    return result