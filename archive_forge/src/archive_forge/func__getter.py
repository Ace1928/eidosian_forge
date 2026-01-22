import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _getter(self):
    """Deprecated. Use value instead."""
    return self.value