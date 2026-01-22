import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class _ABC(type):

    def __init__(cls, *args):
        cls.__doc__ = 'Deprecated AST node class. Use ast.Constant instead'

    def __instancecheck__(cls, inst):
        if not isinstance(inst, Constant):
            return False
        if cls in _const_types:
            try:
                value = inst.value
            except AttributeError:
                return False
            else:
                return isinstance(value, _const_types[cls]) and (not isinstance(value, _const_types_not.get(cls, ())))
        return type.__instancecheck__(cls, inst)