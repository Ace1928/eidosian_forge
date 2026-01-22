import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _dims_getter(self):
    """Deprecated. Use elts instead."""
    return self.elts