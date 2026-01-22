import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def _make_function(code, globals, name, argdefs, closure):
    globals['__builtins__'] = __builtins__
    return types.FunctionType(code, globals, name, argdefs, closure)