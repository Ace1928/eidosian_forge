import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _write_docstring(self, node):
    self.fill()
    if node.kind == 'u':
        self.write('u')
    self._write_str_avoiding_backslashes(node.value, quote_types=_MULTI_QUOTES)