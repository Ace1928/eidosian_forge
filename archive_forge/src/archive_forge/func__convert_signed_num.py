import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _convert_signed_num(node):
    if isinstance(node, UnaryOp) and isinstance(node.op, (UAdd, USub)):
        operand = _convert_num(node.operand)
        if isinstance(node.op, UAdd):
            return +operand
        else:
            return -operand
    return _convert_num(node)