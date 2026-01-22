import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _write_constant(self, value):
    if isinstance(value, (float, complex)):
        self.write(repr(value).replace('inf', _INFSTR).replace('nan', f'({_INFSTR}-{_INFSTR})'))
    elif self._avoid_backslashes and isinstance(value, str):
        self._write_str_avoiding_backslashes(value)
    else:
        self.write(repr(value))