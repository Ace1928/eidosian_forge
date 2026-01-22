import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
class _Code:
    """
    A fake code object, used by L{_Traceback} via L{_Frame}.

    It is intended to have the same API as the stdlib code type to allow
    interoperation with other tools based on that interface.
    """

    def __init__(self, name, filename):
        self.co_name = name
        self.co_filename = filename
        self.co_lnotab = b''
        self.co_firstlineno = 0
        self.co_argcount = 0
        self.co_varnames = []
        self.co_code = b''
        self.co_cellvars = ()
        self.co_consts = ()
        self.co_flags = 0
        self.co_freevars = ()
        self.co_posonlyargcount = 0
        self.co_kwonlyargcount = 0
        self.co_names = ()
        self.co_nlocals = 0
        self.co_stacksize = 0

    def co_positions(self):
        return ((None, None, None, None),)