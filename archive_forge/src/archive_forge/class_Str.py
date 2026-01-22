import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class Str(Constant, metaclass=_ABC):
    _fields = ('s',)
    __new__ = _new