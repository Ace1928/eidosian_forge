import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class AugLoad(expr_context):
    """Deprecated AST node class.  Unused in Python 3."""